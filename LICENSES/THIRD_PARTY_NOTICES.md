# Third-party Notices

## CIGALE v2025.1

The delayed-tau SFH, nebular emission, IGM/redshifting, and Dale et al. (2014)
host-dust paths are JAX ports of or are validated against CIGALE v2025.1.

- Upstream project: `CIGALE`
- Upstream tag: `v2025.1`
- Upstream commit: `29cb909fe2636800b4acdb1dfc7129d8c8494a24`
- Upstream license: `CeCILL v2`
- Local copy of license text: [CeCILL-v2.txt](CeCILL-v2.txt)

Relevant upstream source files include:

- `pcigale/sed_modules/sfhdelayed.py`
- `pcigale/sed_modules/nebular.py`
- `pcigale/sed_modules/redshifting.py`
- `pcigale/sed_modules/dale2014.py`

The CIGALE-derived resources are:

- `resources/nebular/*`
- `resources/templates/dale2014/*`

## GRAHSP / pcigale

Portions of `jaxsedfit` are derived from or closely based on code and data from `GRAHSP` / `pcigale`.

- Upstream project: `GRAHSP`
- Reference commit used by the AGN regressions: `7d35f5232ac9918a785e8dfe75dff693ab246daf`
- Upstream license: `CeCILL v2`
- Local copy of license text: [CeCILL-v2.txt](CeCILL-v2.txt)

Relevant upstream source files include, among others:

- `pcigale/creation_modules/activate.py`
- `pcigale/creation_modules/activategtorus.py`
- `pcigale/creation_modules/activatelines.py`
- `pcigale/creation_modules/biattenuation.py`
- `pcigale/creation_modules/redshifting.py`
- `pcigale/creation_modules/galdale2014.py`

This repository contains JAX/NumPyro ports and modifications of selected model behavior, plus redistributed resource files used by the current supported subset.

## Vendored resource files

The following bundled resource categories in `src/jaxsedfit/resources/` originate from upstream `GRAHSP` resources or associated template bundles used by `GRAHSP`:

- `resources/filters/filter_registry.txt`
- `resources/filters/*`
- `resources/templates/emission_line_table.formatted`

See the README files in those resource directories for per-directory provenance notes.

## Fe II template (PyQSOFit / jaxqsofit)

`resources/templates/fe_uv.txt` and `resources/templates/fe_optical.txt` are the
Fe II templates distributed with PyQSOFit (`src/pyqsofit/`, GPL-3.0) and
jaxqsofit (`src/jaxqsofit/data/`), copied unchanged: Vestergaard & Wilkes (2001),
Salviander et al. (2007) and Tsuzuki et al. (2006) in the UV and Boroson & Green
(1992) in the optical. See `resources/templates/README.md` for hashes and the
open license question.
