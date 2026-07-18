# Paper 3 provenance archive

Large experiment logs and frozen run records are published outside the Git
history. For each paper software release, obtain both of these files from the
GitHub Release assets or the corresponding Zenodo record:

- `symfluence-paper3-provenance-vX.Y.Z.tar.gz`
- `symfluence-paper3-provenance-vX.Y.Z.tar.gz.sha256`

Verify the download before extracting it:

```bash
shasum -a 256 -c symfluence-paper3-provenance-vX.Y.Z.tar.gz.sha256
tar -xzf symfluence-paper3-provenance-vX.Y.Z.tar.gz
```

The archive contains a generated `MANIFEST.json` with the exact SYMFLUENCE
version, Git commit, creation time, and SHA-256 digest of every included file.
It also contains `COVERAGE.md`, which maps experiment records to manuscript
figures and tables.

## Archive source layout

Before packaging, maintainers assemble a staging directory with this layout:

```text
paper3-provenance/
├── COVERAGE.md
├── resolved_configs/
├── run_manifests/
├── curated_logs/
└── reference_metrics/
```

Raw downloaded forcing data and full model-output directories should not be
included. Only curated records needed to substantiate or reproduce reported
results belong in the archive. Confirm that every included dataset may legally
be redistributed and that logs contain no credentials or machine-local secrets.

Create the archive from the repository root:

```bash
scripts/create_paper_provenance_bundle.sh \
  /path/to/paper3-provenance vX.Y.Z dist/paper-provenance
```

The script refuses a release-tag/version mismatch, inventories the files, adds
the generated manifest, builds the tarball, and writes its external checksum.
Upload both outputs to the matching GitHub Release and to the version-specific
Zenodo deposit. Put the final asset links and Zenodo DOI in this file before
the paper release is tagged.

## Release links

- GitHub Release asset: pending
- Zenodo version DOI: pending
