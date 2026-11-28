"""Opt-in empirical UV Fe II template (``FeIITemplate(name="BruhweilerVerner08+EmpiricalUV")``)."""

import numpy as np
import jax
from numpyro.handlers import seed, substitute, trace

from jaxsedfit.config import FeIITemplate, FilterCurve, FilterSet, SpectroscopyData
from jaxsedfit.filters import _package_resource_path
from jaxsedfit.model import grahsp_photometric_model
from jaxsedfit.preload import (
    FEII_EMPIRICAL_UV_RESOURCE,
    FEII_EMPIRICAL_UV_TEMPLATE_NAME,
    _bruhweiler_verner_feii_template,
    _bruhweiler_verner_with_empirical_uv,
    _load_templates,
    build_model_context,
)

from test_model_assembly import _cfg, _patch_ssp


def _integral(wave, flux, lo, hi):
    inside = (wave >= lo) & (wave <= hi)
    return np.trapezoid(flux[inside], wave[inside])


def _hybrid():
    return _bruhweiler_verner_with_empirical_uv(*_bruhweiler_verner_feii_template())


def test_default_template_is_the_unchanged_bruhweiler_verner_table():
    # The pre-existing loader, written out: the default must stay bit-for-bit.
    data = np.loadtxt(_package_resource_path("resources/templates/Fe_d11-m20-20.5.txt"))
    wave_rest = data[:, 0] / (4593.4 / 4575.0)
    llam = data[:, 1] * 2.99792458e18 / np.clip(data[:, 0] * data[:, 0], 1e-30, None)
    llam = llam / llam[int(np.argmin(np.abs(wave_rest - 4575.0)))]

    cfg = _cfg(fit_host=False)
    cfg.agn.feii_template = FeIITemplate()
    loaded = _load_templates(cfg)

    np.testing.assert_array_equal(loaded.feii_wave, wave_rest)
    np.testing.assert_array_equal(loaded.feii_lumin, llam)


def test_empirical_uv_resource_is_the_pyqsofit_table():
    data = np.loadtxt(_package_resource_path(FEII_EMPIRICAL_UV_RESOURCE), comments="#")
    wave = 10.0 ** data[:, 0]
    assert data.shape == (3418, 2)
    assert np.all(np.diff(wave) > 0.0)
    # 103.6 km/s pixels, 1075-3499 A, zero below 1250 A.
    np.testing.assert_allclose(np.median(np.diff(np.log(wave))) * 299792.458, 103.6, atol=0.2)
    assert 1074.0 < wave[0] < 1076.0 and 3498.0 < wave[-1] < 3500.0
    assert np.all(data[wave < 1250.0, 1] == 0.0)
    # Gaps between Fe II multiplets are zero in the table; the 3090-3500 A part has none.
    assert np.mean(data[(wave > 2200.0) & (wave < 3090.0), 1] > 0.0) > 0.95
    assert np.all(data[(wave > 3100.0) & (wave < 3499.0), 1] > 0.0)


def test_empirical_uv_template_normalization_and_coverage():
    bv_wave, bv_lumin = _bruhweiler_verner_feii_template()
    wave, flux = _hybrid()

    assert np.all(np.isfinite(flux)) and np.all(flux >= 0.0)
    assert np.all(np.diff(np.log(wave)) > 0.0)
    np.testing.assert_allclose(np.diff(np.log(wave)), np.median(np.diff(np.log(wave))), rtol=1e-6)
    # Coverage: empirical blue end (zero below 1250 A) to the theoretical red end.
    assert wave[0] < 1100.0 and bv_wave[-1] - 10.0 < wave[-1] <= bv_wave[-1]
    assert np.all(flux[wave < 1250.0] == 0.0)
    assert _integral(wave, flux, 1250.0, 2000.0) > 0.0
    # Same UV Fe II flux over 2200-3090 A as the theoretical template (ratio semantics kept).
    np.testing.assert_allclose(
        _integral(wave, flux, 2200.0, 3090.0), _integral(bv_wave, bv_lumin, 2200.0, 3090.0), rtol=2e-3
    )
    # Optical part: the theoretical template (4575 A normalization kept).
    optical = wave >= 3500.0
    np.testing.assert_allclose(flux[optical], np.interp(wave[optical], bv_wave, bv_lumin), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        _integral(wave, flux, 3600.0, 7000.0), _integral(bv_wave, bv_lumin, 3600.0, 7000.0), rtol=1e-3
    )
    # The shape differs from the theoretical one in the UV (the point of the option).
    assert _integral(wave, flux, 3090.0, 3500.0) > 2.0 * _integral(bv_wave, bv_lumin, 3090.0, 3500.0)


def test_empirical_uv_hand_over_is_continuous():
    bv_wave, bv_lumin = _bruhweiler_verner_feii_template()
    wave, flux = _hybrid()
    emp = np.loadtxt(_package_resource_path(FEII_EMPIRICAL_UV_RESOURCE), comments="#")
    emp_wave, emp_flux = 10.0 ** emp[:, 0], np.clip(emp[:, 1], 0.0, None)
    scale = _integral(bv_wave, bv_lumin, 2200.0, 3090.0) / _integral(emp_wave, emp_flux, 2200.0, 3090.0)

    # Below the hand-over: the scaled empirical template.
    blue = wave <= 3300.0
    np.testing.assert_allclose(flux[blue], scale * np.interp(wave[blue], emp_wave, emp_flux), rtol=1e-3)
    # No step where the empirical template ends: pixel-to-pixel changes across the
    # join stay within the pixel-to-pixel structure of the two templates themselves.
    join = (wave > 3450.0) & (wave < 3560.0)
    step = np.max(np.abs(np.diff(flux[join])))
    level = np.max(flux[(wave > 3300.0) & (wave < 3600.0)])
    assert step < 0.1 * level
    i_end = int(np.searchsorted(wave, emp_wave[-1]))
    assert abs(flux[i_end] - flux[i_end - 1]) < 0.1 * level


def test_empirical_uv_option_is_selected_by_name_and_cached_separately():
    cfg = _cfg(fit_host=False)
    cfg.agn.feii_template = FeIITemplate()
    default = _load_templates(cfg)
    cfg.agn.feii_template = FeIITemplate(name=FEII_EMPIRICAL_UV_TEMPLATE_NAME)
    empirical = _load_templates(cfg)
    wave, flux = _hybrid()

    np.testing.assert_array_equal(empirical.feii_wave, wave)
    np.testing.assert_array_equal(empirical.feii_lumin, flux)
    assert default.feii_wave.size != empirical.feii_wave.size
    np.testing.assert_array_equal(default.line_wave, empirical.line_wave)


def _joint_cfg(name):
    cfg = _cfg(fit_host=False, spectroscopy_enabled=True)
    cfg.agn.fit_lines = False
    cfg.agn.use_smart_line_priors = False
    cfg.agn.fit_feii = True
    cfg.agn.fit_balmer_continuum = True
    cfg.agn.feii_template = FeIITemplate(name=name)
    wave = np.linspace(2400.0, 6000.0, 300)  # rest 2286-5714 A at z = 0.05
    cfg.spectroscopy = SpectroscopyData(wave_obs=wave, fluxes=np.ones_like(wave), errors=0.1 * np.ones_like(wave))
    cfg.filters = FilterSet(curves=[FilterCurve(name="f1", wave=[3000.0, 3400.0, 3800.0], transmission=[0.0, 1.0, 0.0])])
    return cfg


def _joint_model(context):
    return lambda: grahsp_photometric_model(context, include_components=True, include_spectral_features=True)


def test_empirical_uv_option_changes_only_the_feii_component(monkeypatch):
    _patch_ssp(monkeypatch)
    ctx_default = build_model_context(_joint_cfg("BruhweilerVerner08"))
    ctx_empirical = build_model_context(_joint_cfg(FEII_EMPIRICAL_UV_TEMPLATE_NAME))

    tr_default = trace(seed(_joint_model(ctx_default), jax.random.PRNGKey(3))).get_trace()
    latent = {k: v["value"] for k, v in tr_default.items() if v["type"] == "sample" and not v.get("is_observed")}
    tr_empirical = trace(seed(substitute(_joint_model(ctx_empirical), data=latent), jax.random.PRNGKey(3))).get_trace()

    # Same sites and priors: the option adds no parameter.
    assert set(tr_default) == set(tr_empirical)
    for name in latent:
        assert tr_empirical[name]["type"] == "sample"
        np.testing.assert_array_equal(tr_empirical[name]["value"], latent[name])
        np.testing.assert_array_equal(
            tr_default[name]["fn"].log_prob(latent[name]), tr_empirical[name]["fn"].log_prob(latent[name])
        )

    def value(tr, name):
        return np.asarray(tr[name]["value"])

    # Fe II itself changes, in the spectrum, the photometry and the SED rendering.
    for name in ("spectral_feii_model", "spectral_feii_photometry", "spectral_feii_obs_sed"):
        assert not np.allclose(value(tr_default, name), value(tr_empirical, name), rtol=1e-3, atol=0.0)
    # Every other component is identical.
    for name in ("spectral_continuum_model", "spectral_balmer_model", "spectral_balmer_photometry", "spectral_balmer_obs_sed", "disk_obs_sed", "torus_obs_sed"):
        np.testing.assert_array_equal(value(tr_default, name), value(tr_empirical, name))
    # The total is the sum of its parts, so the Fe II change appears once (no double counting).
    delta_total = value(tr_empirical, "spectral_total_model") - value(tr_default, "spectral_total_model")
    delta_feii = value(tr_empirical, "spectral_feii_model") - value(tr_default, "spectral_feii_model")
    np.testing.assert_allclose(delta_total, delta_feii, rtol=1e-10, atol=1e-12 * np.max(np.abs(value(tr_default, "spectral_total_model"))))
    delta_sed = value(tr_empirical, "agn_obs_sed") - value(tr_default, "agn_obs_sed")
    delta_feii_sed = value(tr_empirical, "spectral_feii_obs_sed") - value(tr_default, "spectral_feii_obs_sed")
    np.testing.assert_allclose(delta_sed, delta_feii_sed, rtol=1e-10, atol=1e-12 * np.max(np.abs(value(tr_default, "agn_obs_sed"))))
    # The optical part keeps the theoretical template: Fe II redward of the hand-over agrees.
    rest = np.asarray(ctx_default.spec_wave_obs) / 1.05
    optical = rest > 4200.0
    np.testing.assert_allclose(
        value(tr_empirical, "spectral_feii_model")[optical], value(tr_default, "spectral_feii_model")[optical], rtol=2e-2,
        atol=1e-3 * np.max(value(tr_default, "spectral_feii_model")),
    )


def test_empirical_uv_2200_variant_changes_only_2200_to_3500():
    """The 2200 variant equals the theoretical template below 2100 A and above 3500 A and the full
    empirical-UV template between 2200 and 3300 A."""
    from jaxsedfit import preload

    bv_w, bv_l = preload._bruhweiler_verner_feii_template()
    w_full, l_full = preload._bruhweiler_verner_with_empirical_uv(bv_w, bv_l)
    w_cut, l_cut = preload._bruhweiler_verner_with_empirical_uv(
        bv_w, bv_l, blue_taper=preload.FEII_EMPIRICAL_UV_BLUE_TAPER_A
    )
    np.testing.assert_array_equal(w_full, w_cut)
    theo = np.interp(w_cut, bv_w, np.clip(bv_l, 0.0, None), left=0.0, right=0.0)
    blue, mid, red = w_cut < 2100.0, (w_cut > 2200.0) & (w_cut < 3300.0), w_cut > 3500.0
    np.testing.assert_allclose(l_cut[blue], theo[blue], atol=1e-12)
    np.testing.assert_allclose(l_cut[mid], l_full[mid], rtol=1e-12)
    np.testing.assert_allclose(l_cut[red], theo[red], atol=1e-12)
    assert float(np.sum(l_cut[w_cut < 1990.0])) == 0.0
    assert float(np.sum(l_full[w_full < 1990.0])) > 0.0
