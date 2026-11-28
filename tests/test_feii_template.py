"""The only built-in Fe II template is the jaxqsofit UV + optical composite.

The vendored files ``fe_uv.txt`` and ``fe_optical.txt`` are copied unchanged
from jaxqsofit. jaxqsofit reads both as rest-frame f_lambda with log10
wavelengths in column 0, clips negative values, rescales each piece so its
99th-percentile amplitude is 1, and keeps the optical piece at 3686-7484 A.
jaxsedfit does the same and then multiplies both pieces by one common factor,
chosen so that the mean of the template over the 4434-4684 A optical Fe II band
keeps the value of the previous default template. ``feii_norm`` therefore keeps
its meaning as the optical Fe II strength, and the UV-to-optical ratio at
``log_feii_op_over_uv = 0`` is the jaxqsofit one.
"""

from hashlib import sha256
from importlib import resources
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from jaxsedfit.config import AGNConfig, FeIITemplate, FilterSet
from jaxsedfit.preload import (
    FEII_OPTICAL_BAND_A,
    FEII_OPTICAL_BAND_MEAN,
    FEII_TEMPLATE_NAME,
    _TEMPLATE_CACHE,
    _load_templates,
    build_model_context,
)
from jaxsedfit.spectral_model import FEII_UV_OPTICAL_SPLIT_A, _fe_template_component, feii_optical_scale
from jaxsedfit.spectroscopy import SpectralComponentConfig, evaluate_joint_spectral_components

from test_model_assembly import _cfg, _deterministic_trace, _fixed_component_data, _patch_ssp, _site

_JAXQSOFIT_FILE_HASHES = {
    # jaxqsofit commit 1e75ce5, src/jaxqsofit/data/
    "fe_uv.txt": "f2bbdd82c6c66337f61e858fe7abd0c9666eee586818fc9db1e06567a659eb7d",
    "fe_optical.txt": "096c6ed6ca2f97a401ffca24b2d8577d46c699d06fe77569cb491aa15a6ee300",
}


def _resource(name):
    return resources.files("jaxsedfit").joinpath("resources/templates", name)


def _jaxqsofit_pieces():
    """Reproduce jaxqsofit's own loading of the two template pieces."""

    def _norm99(flux):
        flux = np.maximum(np.asarray(flux, dtype=float), 0.0)
        return flux / np.nanpercentile(np.abs(flux), 99)

    uv = np.genfromtxt(str(_resource("fe_uv.txt")))
    op = np.genfromtxt(str(_resource("fe_optical.txt")))
    uv_wave, uv_flux = 10.0 ** uv[:, 0], _norm99(uv[:, 1])
    op_wave, op_flux = 10.0 ** op[:, 0], _norm99(op[:, 1])
    keep = (op_wave > 3686.0) & (op_wave < 7484.0)
    return uv_wave, uv_flux, op_wave[keep], op_flux[keep]


def _band_mean(wave, flux, lo, hi):
    grid = np.union1d(wave[(wave > lo) & (wave < hi)], [lo, hi])
    return np.trapezoid(np.interp(grid, wave, flux), grid) / (hi - lo)


def _default_templates():
    _TEMPLATE_CACHE.clear()
    holder = SimpleNamespace(
        agn=AGNConfig(fit_agn=True),
        spectroscopy_list=[],
        galaxy=SimpleNamespace(fit_host=False, use_energy_balance=False, dust_alpha=2.0),
    )
    return _load_templates(holder)


def test_vendored_feii_files_are_unchanged_jaxqsofit_copies():
    for name, expected in _JAXQSOFIT_FILE_HASHES.items():
        assert sha256(_resource(name).read_bytes()).hexdigest() == expected


def test_old_grahsp_feii_file_is_no_longer_shipped():
    assert not _resource("Fe_d11-m20-20.5.txt").is_file()


def test_default_feii_template_name_is_jaxqsofit():
    assert FEII_TEMPLATE_NAME == "jaxqsofit"
    assert FeIITemplate().name == "jaxqsofit"


def test_default_template_is_the_jaxqsofit_composite_with_one_common_scale():
    templates = _default_templates()
    wave, flux = templates.feii_wave, templates.feii_lumin
    uv_wave, uv_flux, op_wave, op_flux = _jaxqsofit_pieces()

    assert np.all(np.diff(wave) > 0.0)
    assert np.all(flux >= 0.0)
    # One common factor on both pieces: exactly the jaxqsofit values elsewhere.
    scale = FEII_OPTICAL_BAND_MEAN / _band_mean(op_wave, op_flux, *FEII_OPTICAL_BAND_A)
    np.testing.assert_allclose(np.interp(uv_wave, wave, flux), scale * uv_flux, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(np.interp(op_wave, wave, flux), scale * op_flux, rtol=1.0e-12, atol=0.0)
    # No flux between the two pieces (jaxqsofit interpolates each with left=right=0).
    gap = (wave > uv_wave[-1]) & (wave < op_wave[0])
    assert np.any(gap)
    assert np.all(flux[gap] == 0.0)
    assert wave[0] == pytest.approx(uv_wave[0], rel=1.0e-12)
    assert wave[-1] == pytest.approx(op_wave[-1], rel=1.0e-12)


def test_feii_norm_keeps_its_optical_meaning():
    templates = _default_templates()
    # The previous default template had this mean over 4434-4684 A.
    assert FEII_OPTICAL_BAND_A == (4434.0, 4684.0)
    assert FEII_OPTICAL_BAND_MEAN == pytest.approx(0.274125, abs=5.0e-7)
    assert _band_mean(templates.feii_wave, templates.feii_lumin, *FEII_OPTICAL_BAND_A) == pytest.approx(
        FEII_OPTICAL_BAND_MEAN, rel=1.0e-10
    )
    # The UV-to-optical ratio is the jaxqsofit one (each piece at unit 99th percentile).
    uv_wave, uv_flux, op_wave, op_flux = _jaxqsofit_pieces()
    expected_ratio = _band_mean(uv_wave, uv_flux, 2200.0, 3090.0) / _band_mean(op_wave, op_flux, 4434.0, 4684.0)
    ratio = _band_mean(templates.feii_wave, templates.feii_lumin, 2200.0, 3090.0) / FEII_OPTICAL_BAND_MEAN
    assert ratio == pytest.approx(expected_ratio, rel=1.0e-10)


def test_uv_optical_split_falls_between_the_two_pieces():
    templates = _default_templates()
    wave, flux = templates.feii_wave, templates.feii_lumin
    uv_wave, uv_flux, op_wave, op_flux = _jaxqsofit_pieces()
    assert uv_wave[-1] < FEII_UV_OPTICAL_SPLIT_A < op_wave[0]

    scaled = np.asarray(flux * feii_optical_scale(wave, np.log(2.5)))
    blue, red = wave < FEII_UV_OPTICAL_SPLIT_A, wave >= FEII_UV_OPTICAL_SPLIT_A
    np.testing.assert_allclose(scaled[blue], flux[blue], rtol=1.0e-12)
    np.testing.assert_allclose(scaled[red], 2.5 * flux[red], rtol=1.0e-12)


def test_shared_amplitude_matches_jaxqsofit_separate_pieces():
    """Broadening the combined template equals broadening the two jaxqsofit pieces."""
    templates = _default_templates()
    uv_wave, uv_flux, op_wave, op_flux = _jaxqsofit_pieces()
    scale = FEII_OPTICAL_BAND_MEAN / _band_mean(op_wave, op_flux, *FEII_OPTICAL_BAND_A)
    wave = np.geomspace(1300.0, 7300.0, 4000)
    norm, log_ratio, fwhm, shift = 2.0, np.log(0.6), 3000.0, 0.001

    combined = _fe_template_component(
        wave,
        templates.feii_wave,
        templates.feii_lumin * feii_optical_scale(templates.feii_wave, log_ratio),
        norm,
        fwhm,
        shift,
    )
    # jaxqsofit: Fe_op_norm = Fe_uv_norm * exp(log_Fe_op_over_uv); here with shared width and shift.
    separate = _fe_template_component(wave, uv_wave, scale * uv_flux, norm, fwhm, shift) + _fe_template_component(
        wave, op_wave, scale * op_flux, norm * np.exp(log_ratio), fwhm, shift
    )
    combined, separate = np.asarray(combined), np.asarray(separate)
    assert np.max(np.abs(combined - separate)) < 2.0e-3 * np.max(separate)


def test_spectral_feii_component_reproduces_the_template_flambda_shape():
    """Units round trip: the mJy Fe II model divided by lambda^2 is the template f_lambda shape."""
    templates = _default_templates()
    wave_obs = np.geomspace(1500.0, 7000.0, 3000)
    fwhm = 2500.0
    fn = substitute(
        seed(evaluate_joint_spectral_components, jax.random.PRNGKey(3)),
        data={
            "spectral_feii_norm": 1.0,
            "spectral_feii_fwhm": fwhm,
            "spectral_feii_shift": 0.0,
            "spectral_feii_log_op_over_uv": 0.0,
        },
    )
    tr = trace(fn).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=np.zeros_like(wave_obs),
        config=SpectralComponentConfig(
            use_lines=False, use_feii=True, use_balmer_continuum=False, feii_fnu_pivot_rest=4575.0
        ),
        feii_template_wave_rest=templates.feii_wave,
        feii_template_flux=templates.feii_lumin,
    )
    feii_fnu = np.asarray(tr["spectral_feii_model"]["value"])
    expected_flambda = np.asarray(
        _fe_template_component(wave_obs, templates.feii_wave, templates.feii_lumin, 1.0, fwhm, 0.0)
    )
    flambda = feii_fnu / wave_obs**2
    strong = expected_flambda > 0.05 * expected_flambda.max()
    ratio = flambda[strong] / expected_flambda[strong]
    np.testing.assert_allclose(ratio, np.median(ratio), rtol=1.0e-6)


def _contexts(monkeypatch):
    _patch_ssp(monkeypatch)
    _TEMPLATE_CACHE.clear()

    def _default_feii(cfg):
        cfg.agn.feii_template = FeIITemplate()
        return cfg

    photometry = _default_feii(_cfg(fit_host=False, n_wave=512))
    spectrum = _default_feii(_cfg(spectroscopy_enabled=True, fit_host=False, n_wave=256))
    spectrum.photometry = None
    spectrum.filters = FilterSet()
    joint = _default_feii(_cfg(spectroscopy_enabled=True, fit_host=False, n_wave=256))
    for cfg in (spectrum, joint):
        cfg.agn.fit_lines = False
        cfg.agn.use_smart_line_priors = False
    return {name: (cfg, build_model_context(cfg)) for name, cfg in
            (("photometry", photometry), ("spectrum", spectrum), ("joint", joint))}


def test_all_three_modes_load_the_jaxqsofit_template(monkeypatch):
    reference = _default_templates()
    for name, (cfg, context) in _contexts(monkeypatch).items():
        assert cfg.agn.feii_template.name == "jaxqsofit", name
        np.testing.assert_array_equal(context.templates.feii_wave, reference.feii_wave)
        np.testing.assert_array_equal(context.templates.feii_lumin, reference.feii_lumin)
        expected = np.interp(context.rest_wave, reference.feii_wave, reference.feii_lumin, left=0.0, right=0.0)
        np.testing.assert_allclose(np.asarray(context.feii_template_on_rest_jax), expected, rtol=1.0e-12, atol=0.0)


def test_native_sed_feii_is_the_template_times_the_amplitude(monkeypatch):
    _, context = _contexts(monkeypatch)["photometry"]
    tr = _deterministic_trace(context, _fixed_component_data())
    feii = _site(tr, "feii_rest_sed")
    template = np.asarray(context.feii_template_on_rest_jax)
    op = np.where(np.asarray(context.rest_wave) >= FEII_UV_OPTICAL_SPLIT_A, np.exp(float(_site(tr, "log_feii_op_over_uv"))), 1.0)
    shape = template * op
    nonzero = shape > 0.0
    assert np.any(nonzero)
    ratio = feii[nonzero] / shape[nonzero]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1.0e-10)


def test_joint_spectral_and_native_feii_use_the_same_template(monkeypatch):
    import jaxsedfit.model as model_module

    contexts = _contexts(monkeypatch)
    _, context = contexts["joint"]
    seen = []
    real = model_module._evaluate_spectral_components

    def _spy(wave_obs, redshift, continuum_mjy, cfg, prior_config, feii_wave, feii_flux, *args, **kwargs):
        seen.append((np.asarray(feii_wave), np.asarray(feii_flux)))
        return real(wave_obs, redshift, continuum_mjy, cfg, prior_config, feii_wave, feii_flux, *args, **kwargs)

    monkeypatch.setattr("jaxsedfit.model._evaluate_spectral_components", _spy)
    _deterministic_trace(context, _fixed_component_data())
    assert seen
    for feii_wave, feii_flux in seen:
        np.testing.assert_array_equal(feii_wave, context.templates.feii_wave)
        np.testing.assert_array_equal(feii_flux, context.templates.feii_lumin)


def test_removed_template_name_raises_a_clear_error(monkeypatch):
    _patch_ssp(monkeypatch)
    _TEMPLATE_CACHE.clear()
    cfg = _cfg(fit_host=False, n_wave=256)
    cfg.agn.feii_template = FeIITemplate(name="BruhweilerVerner08")
    with pytest.raises(ValueError, match="BruhweilerVerner08.*removed.*jaxqsofit"):
        build_model_context(cfg)
    with pytest.raises(ValueError, match="BruhweilerVerner08.*removed.*jaxqsofit"):
        AGNConfig(feii_template=FeIITemplate(name="BruhweilerVerner08")).validate()


def test_unknown_template_name_still_raises(monkeypatch):
    _patch_ssp(monkeypatch)
    _TEMPLATE_CACHE.clear()
    cfg = _cfg(fit_host=False, n_wave=256)
    cfg.agn.feii_template = FeIITemplate(name="NoSuchTemplate")
    with pytest.raises(ValueError, match="Unsupported Fe II template"):
        build_model_context(cfg)


def test_inline_template_arrays_are_still_accepted_under_any_name(monkeypatch):
    _patch_ssp(monkeypatch)
    _TEMPLATE_CACHE.clear()
    cfg = _cfg(fit_host=False, n_wave=256)
    cfg.agn.feii_template = FeIITemplate(
        name="BruhweilerVerner08", wave=[1000.0, 2000.0], lumin=[1.0, 0.5], wavelength_unit="angstrom"
    )
    context = build_model_context(cfg)
    np.testing.assert_array_equal(context.templates.feii_wave, [1000.0, 2000.0])
