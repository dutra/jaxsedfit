# Template provenance

This directory contains vendored template files redistributed with `jaxsedfit`.

## From GRAHSP

- Upstream project: `GRAHSP`
- Upstream license: `CeCILL v2`
- Local license copy: [CeCILL-v2.txt](../../../../LICENSES/CeCILL-v2.txt)

Vendored files:

- `emission_line_table.formatted`
  Source: `GRAHSP/database_builder/activate/agn/mor_netzer_2012/emission_line_table.formatted`

The `dale2014/` subdirectory has its own provenance note.

## Fe II template (jaxqsofit / PyQSOFit)

- `fe_uv.txt`, `fe_optical.txt`
  The Fe II templates used by jaxqsofit (`src/jaxqsofit/data/`, commit
  `1e75ce5`), copied unchanged. They are byte-identical to the PyQSOFit files
  `src/pyqsofit/fe_uv.txt` and `src/pyqsofit/fe_optical.txt` (PyQSOFit commit
  `084df5a`; Guo, Shen & Wang 2018; Shen et al. 2019).
  SHA-256: `fe_uv.txt`
  `f2bbdd82c6c66337f61e858fe7abd0c9666eee586818fc9db1e06567a659eb7d`,
  `fe_optical.txt`
  `096c6ed6ca2f97a401ffca24b2d8577d46c699d06fe77569cb491aa15a6ee300`.
  - `fe_uv.txt`: Vestergaard & Wilkes (2001) at 1000-2200 A, Salviander et
    al. (2007) at 2200-3090 A (extrapolated under Mg II), Tsuzuki et al. (2006)
    at 3090-3500 A (from the file header). Columns: log10 rest wavelength [A]
    and f_lambda [erg/s/cm2/A]; 103.6 km/s pixels; zero below 1250 A.
  - `fe_optical.txt`: Boroson & Green (1992) optical template. Columns: log10
    rest wavelength [A] and f_lambda [erg/s/cm2/A]; 106 km/s pixels.
  - How they are combined and normalized: `_load_vendored_feii_template` in
    `preload.py` and the Fe II template section of `docs/agn.rst`.
  - License: PyQSOFit is distributed under GPL-3.0. jaxqsofit ships no
    license file. The templates themselves are published literature data;
    whether their redistribution here is compatible with the jaxsedfit
    license has not been checked and should be before a release.
