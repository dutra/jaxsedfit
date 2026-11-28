# Template provenance

This directory contains vendored template files redistributed with `jaxsedfit`.

- Upstream project: `GRAHSP`
- Upstream license: `CeCILL v2`
- Local license copy: [CeCILL-v2.txt](../../../../LICENSES/CeCILL-v2.txt)

Vendored files:

- `Fe_d11-m20-20.5.txt`
  Source: `GRAHSP/database_builder/activate/agn/FeII_template/Fe_d11-m20-20.5.txt`
- `emission_line_table.formatted`
  Source: `GRAHSP/database_builder/activate/agn/mor_netzer_2012/emission_line_table.formatted`

The `dale2014/` subdirectory has its own provenance note.

Not from GRAHSP:

- `fe_uv_VW01_S07_T06.txt`
  The empirical UV Fe II template distributed with PyQSOFit (Shen et al. 2011;
  Guo, Shen & Wang 2018) and used by its UV Fe II fits, copied unchanged from
  jaxqsofit (`src/jaxqsofit/data/fe_uv.txt`; SHA-256
  `f2bbdd82c6c66337f61e858fe7abd0c9666eee586818fc9db1e06567a659eb7d`). Its
  header gives the composition: Vestergaard & Wilkes (2001) at 1000-2200 A,
  Salviander et al. (2007) at 2200-3090 A (extrapolated under Mg II), Tsuzuki
  et al. (2006) at 3090-3500 A; columns log10 rest wavelength [A] and f_lambda,
  103.6 km/s pixels, zero below 1250 A. Used only by the opt-in
  `FeIITemplate(name="BruhweilerVerner08+EmpiricalUV")`, which replaces the UV
  part of `Fe_d11-m20-20.5.txt` by this template (raised-cosine hand-over over
  3300-3500 A, same flux over 2200-3090 A). Licence: jaxqsofit ships no
  licence for this file and the redistribution terms of the PyQSOFit template
  were not verified; check before a public release.
