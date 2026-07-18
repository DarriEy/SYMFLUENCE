# Paper 3 provenance coverage

This tracked inventory defines the records expected in the external provenance
archive. Before publishing the paper release, replace each `pending` entry with
the corresponding archive paths and confirm that the reported reference
metrics agree with the manuscript.

| Experiment | Manuscript output | Required archive records | Status |
|---|---|---|---|
| 01 domain definition | Figures 1–3; Table 1 | resolved configs, run manifests, curated setup logs | pending |
| 02 model ensemble | Figure 7 | resolved configs and manifests for all 17 model runs; evaluation metrics | pending |
| 03 forcing ensemble | Figure 6 | four forcing-run records and comparison metrics | pending |
| 04 calibration ensemble | Figure 9 | calibration manifests, curated optimizer logs, consolidated metrics | pending |
| 05 benchmarking | Figure 8 | benchmark manifest, scores, and reference metrics | pending |
| 06 multivariate evaluation | Figure 10 | four experiment manifests and streamflow/TWS metrics | pending |
| 07 data pipeline | Figure 4 | pipeline manifest and curated transformation log | pending |
| 08 parallel scaling | Figure 11 | platform manifests, timing summaries, and scaling metrics | pending |

The figure-generation inputs themselves are described in
[`../plotting/README.md`](../plotting/README.md). Raw forcing data and complete
model-output directories are intentionally outside the provenance archive.
