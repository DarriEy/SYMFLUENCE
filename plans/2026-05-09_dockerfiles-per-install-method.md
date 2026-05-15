# Plan: Dockerfiles per supported installation method

**Goal:** produce a separate Dockerfile for each SYMFLUENCE installation method documented in
`README.md` (lines 28–94) and on https://symfluence.readthedocs.io/en/latest/installation.html ,
so users can `docker build` an image that mirrors any documented install path.

**Scope decisions made up front:**
- HPC module-load recipes (Anvil, ARC, FIR) are environment-specific and intentionally **out of
  scope** — they are not container-friendly.
- Windows-native installs are **out of scope** for Docker (Linux containers only).
- The existing top-level `Dockerfile` is treated as the **conda/source reference template** —
  Phase 6 and Phase 7 will re-use its multi-stage layout.

**Two-file convention (every phase):**
- `docker/<method>/Dockerfile` = the base install method **as described in the docs**, verbatim.
  No workarounds. If the docs say `pip install symfluence` + `symfluence binary install`, that's
  what the Dockerfile does — even if it produces a partially-broken image.
- `docker/<method>/Dockerfile.fixed` = a copy of `Dockerfile` with the **minimum set of
  workarounds** needed to produce a fully-working image. Each fix gets a comment that names the
  upstream bug it papers over, so a future maintainer can drop the workaround when it's fixed
  upstream. Only create `Dockerfile.fixed` if the base build actually has a problem worth fixing.

**Known fixes from Phase 2 (pip) — likely reusable for uv, uv-tool, pipx, source phases**
(these all share the `pip install symfluence` + `symfluence binary install` flow on
`python:3.11-slim-bookworm`):
1. apt: `+ libproj-dev libgeos-dev libudunits2-dev pkg-config wget`
2. Symlink `/usr/lib/<arch>/libhdf5_serial*.so` → `/usr/lib/libhdf5*.so` (note Debian quirk:
   `libhdf5_serialhl_fortran.so` has no underscore between "serial" and "hl")
3. Symlink real libc to `/lib64/libc.so.6` (works around `_build.sh` x86_64-only host-libc probe)
4. Hide `/usr/include/boost` between `pip install` and `symfluence binary install`
   (libboost-dev 1.74 is needed for some pip wheels but breaks ngen's `find_package(Boost ≥ 1.79)`)

The `npm` (pre-built binaries) and `conda` (different dep resolver) phases follow different code
paths; they may not need any of these fixes — assess from the base build before writing `.fixed`.

---

## Phase 0 — Documentation Discovery (done — keep this section as a session-portable fact sheet)

### Sources of truth
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/README.md` (Installation section, ~lines 28–148)
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/pyproject.toml` (`project.optional-dependencies`, `requires-python = ">=3.11,<3.14"`)
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/docs/source/installation.rst`
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/docs/SYSTEM_REQUIREMENTS.md`
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/Dockerfile` (existing multi-stage build, reference)
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/docker-compose.yaml`
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/.dockerignore`
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/npm/package.json`, `tools/npm/README.md`
- `/Users/lindsayparker/dev/repos/SYMFLUENCE/.github/workflows/install-validate.yml`,
  `cross-platform.yml`, `install-validate-parallel.yml`
- https://symfluence.readthedocs.io/en/latest/installation.html
  (corollary pages `getting_started.html`, `troubleshooting.html`; note `docker.html` is **404** —
  there is no upstream Docker doc to mirror)

### Allowed install commands (verbatim — do not paraphrase)

**Method A — npm global (pre-built binaries):**
```
npm install -g symfluence
symfluence binary info
symfluence binary doctor
```
Supports Linux x86_64 (Ubuntu 22.04+/RHEL 9+/Debian 12+) and macOS ARM64 only.

**Method B — pip (PyPI, then build binaries):**
```
pip install symfluence
symfluence binary install
```

**Method C — uv (env-installed):**
```
uv pip install symfluence
```

**Method D — uv (isolated CLI tool):**
```
uv tool install symfluence
```

**Method E — pipx (isolated CLI):**
```
pipx install symfluence
```

**Method F — conda/mamba (Windows path; also macOS Apple-Silicon GDAL workaround):**
```
conda create -n symfluence python=3.11
conda activate symfluence
conda install -c conda-forge gdal geopandas rasterio netcdf4 hdf5
pip install symfluence
```

**Method G — Source / development bootstrap:**
```
git clone https://github.com/symfluence-org/SYMFLUENCE.git
cd SYMFLUENCE
./scripts/symfluence-bootstrap --install
source venv/bin/activate
```
Manual fallback if bootstrap is not used:
```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### System prerequisites (verbatim, Debian/Ubuntu — the only OS family relevant for Docker)

GDAL build prerequisites (README.md lines 100–105):
```
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

Core libraries (README.md lines 121–124):
```
sudo apt-get install -y libnetcdf-dev libhdf5-dev libproj-dev libgeos-dev
```

Build toolchain (`docs/SYSTEM_REQUIREMENTS.md` lines 156–160):
```
sudo apt-get install -y build-essential gfortran cmake \
    libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev \
    libnetcdf-dev libnetcdff-dev libhdf5-dev
```

Runtime-only libraries for pre-built binaries (`docs/SYSTEM_REQUIREMENTS.md` lines 34–44):
```
sudo apt-get install -y \
    libnetcdf19 libnetcdff7 libhdf5-103 \
    libgdal32 libproj25 libgeos3.11.1 \
    libopenmpi3
```

Optional R support (README.md lines 137–139):
```
sudo apt-get install -y r-base r-base-dev
```

### pyproject.toml extras (verbatim names)
`gdal`, `baseflow`, `r`, `dev`, `test`, `docs`, `notebook`, `gui`, `tui`, `alos`, `hpc`, `all`.

> Note: extras are **not** documented on readthedocs. Treat `pyproject.toml` as authoritative for
> their existence. Default Dockerfiles install no extras unless the method explicitly needs one.

### Anti-patterns to refuse
- Inventing `pip install symfluence[docker]` or any extras group not listed above.
- Inventing `symfluence install-binaries` / `symfluence install` — the only documented binary
  command is `symfluence binary install` (and the diagnostics `symfluence binary info` /
  `symfluence binary doctor`).
- Assuming a `Dockerfile` for npm needs the build toolchain — npm distributes pre-built binaries;
  it needs **runtime libraries only** (the second apt list above), not `build-essential` /
  `gfortran` / `cmake`.
- Calling `./scripts/symfluence-bootstrap` with any flag other than `--install` (no other flag is
  documented on readthedocs).
- Activating `venv/` inside a Dockerfile by sourcing it across `RUN` layers — each `RUN` is a fresh
  shell. Use `ENV PATH=/opt/venv/bin:$PATH` instead.

### Confidence + gaps
- High confidence on commands; all quoted verbatim from README.md, installation.rst, or
  installation.html.
- Gap: per-model docs pages (`models/model_summa.html`, etc.) were not exhaustively read, but the
  upstream pattern is to delegate model-binary builds to `symfluence binary install`, so this is
  unlikely to affect Dockerfiles.
- Gap: behaviour of `symfluence binary install` when network access is restricted is not
  documented — Phase 9 verification needs to confirm builds work in a clean container with only
  the listed apt packages.

---

## Phase 1 — Shared infrastructure

**What to implement (copy-based):**
1. Create directory `docker/` at repo root with one subdirectory per method:
   `docker/pip/`, `docker/uv/`, `docker/uv-tool/`, `docker/pipx/`, `docker/npm/`,
   `docker/conda/`, `docker/source/`. Each holds a single `Dockerfile`.
2. Copy `/Users/lindsayparker/dev/repos/SYMFLUENCE/.dockerignore` review: confirm it excludes
   `docker/`-build artifacts; it currently excludes `Dockerfile` itself, which is fine because
   each new Dockerfile will be passed via `-f docker/<method>/Dockerfile`. **Do not rename or
   relocate the existing root `Dockerfile`** — Phase 8 may keep it as the default.
3. Establish image-tag convention to use in build instructions:
   `symfluence:<method>` (e.g. `symfluence:pip`, `symfluence:npm`, `symfluence:source`).

**Documentation references to follow:**
- Existing `Dockerfile` (multi-stage layout, ENV vars, ENTRYPOINT/CMD) — copy structure.
- `.dockerignore` — do not modify unless a Phase 2–7 verification step proves it's needed.

**Verification checklist:**
- `ls docker/` shows seven subdirectories.
- `docker build -f docker/pip/Dockerfile -t symfluence:pip .` from the repo root resolves the
  build context (no `not found` errors).
- The existing root `Dockerfile` and `docker-compose.yaml` are unchanged.

**Anti-pattern guards:**
- Do not introduce a top-level `Dockerfile.pip` etc. — keep them under `docker/<method>/` so the
  root stays clean.
- Do not move the existing `Dockerfile`; some user workflows reference it directly.

---

## Phase 2 — `docker/pip/Dockerfile` (Method B)

**What to implement (copy-based):**
1. Base image: `python:3.11-slim-bookworm` (matches `requires-python = ">=3.11,<3.14"` and
   Debian 12 — supported by the runtime-libs apt list).
2. As a single `RUN` (with `--mount=type=cache,target=/var/cache/apt`), install the **build
   toolchain** apt list verbatim from `SYSTEM_REQUIREMENTS.md` lines 156–160 plus the GDAL build
   prerequisites from README.md lines 100–105 plus `git curl ca-certificates`.
3. Set the GDAL env vars exactly as in README.md lines 103–104:
   `ENV CPLUS_INCLUDE_PATH=/usr/include/gdal` and `ENV C_INCLUDE_PATH=/usr/include/gdal`.
4. Install the package and binaries with the verbatim commands from
   installation.html "Option 2: pip (Python Framework Only)":
   ```
   RUN pip install --no-cache-dir symfluence
   RUN symfluence binary install
   ```
5. `ENTRYPOINT ["symfluence"]`, `CMD ["--help"]` — copy from existing root `Dockerfile`
   lines that set entrypoint/cmd.

**Documentation references:**
- README.md lines 34, 51–54 (`pip install symfluence` + `symfluence binary install`).
- README.md lines 100–105 (GDAL env vars).
- `docs/SYSTEM_REQUIREMENTS.md` lines 156–160 (apt build toolchain).
- Existing root `Dockerfile` ENTRYPOINT/CMD lines.

**Verification checklist:**
- `docker build -f docker/pip/Dockerfile -t symfluence:pip .` succeeds.
- `docker run --rm symfluence:pip --help` prints SYMFLUENCE help.
- `docker run --rm symfluence:pip binary doctor` runs without raising (warnings allowed).
- `docker run --rm --entrypoint pip symfluence:pip show symfluence` shows the installed version.

**Anti-pattern guards:**
- Do **not** pass any `[extras]` to `pip install` — none are documented on readthedocs for the
  user-facing path.
- Do **not** copy local source into the image — this is the PyPI install path.
- Do **not** install the runtime-only libs alongside the build toolchain; the build toolchain
  superset already covers what's needed.

---

## Phase 3 — `docker/uv/Dockerfile` (Method C: env-installed uv)

**What to implement (copy-based):**
1. Base image: copy Phase 2 base `python:3.11-slim-bookworm`.
2. Copy the apt block + GDAL ENV vars from Phase 2 verbatim.
3. Install uv via the official static binary — use the `ghcr.io/astral-sh/uv:latest` `COPY --from`
   pattern (uv's documented Docker idiom):
   ```
   COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
   ```
4. Install SYMFLUENCE with the verbatim README.md line 41 command:
   ```
   RUN uv pip install --system --no-cache symfluence
   RUN symfluence binary install
   ```
   (`--system` is required because the slim image has no active venv; this matches `uv pip
   install` semantics.)
5. ENTRYPOINT/CMD: copy from Phase 2.

**Documentation references:**
- README.md line 41 (`uv pip install symfluence`).
- README.md line 51–54 (`symfluence binary install`).

**Verification checklist:**
- `docker build -f docker/uv/Dockerfile -t symfluence:uv .` succeeds.
- `docker run --rm --entrypoint uv symfluence:uv --version` prints a version.
- `docker run --rm symfluence:uv --help` prints SYMFLUENCE help.

**Anti-pattern guards:**
- Do not `pip install uv` — use the upstream static binary.
- Do not omit `--system` — without an active venv, uv refuses to install otherwise.
- Do not switch to `uv tool install` here; that's Phase 4.

---

## Phase 4 — `docker/uv-tool/Dockerfile` (Method D: isolated CLI via uv tool)

**What to implement (copy-based):**
1. Base image, apt block, GDAL ENV vars: copy from Phase 3.
2. uv binary install: copy from Phase 3.
3. Install with the verbatim README.md line 43 command:
   ```
   RUN uv tool install symfluence
   ENV PATH=/root/.local/bin:$PATH
   RUN symfluence binary install
   ```
4. ENTRYPOINT/CMD: copy from Phase 2.

**Documentation references:**
- README.md line 43 (`uv tool install symfluence`).

**Verification checklist:**
- `docker build -f docker/uv-tool/Dockerfile -t symfluence:uv-tool .` succeeds.
- `docker run --rm symfluence:uv-tool --help` prints SYMFLUENCE help.
- `docker run --rm --entrypoint uv symfluence:uv-tool tool list` lists `symfluence`.

**Anti-pattern guards:**
- Do not skip the `PATH` update — `uv tool` installs into `~/.local/bin`, which is not on PATH by
  default in slim images.
- Do not combine this with Phase 3's `uv pip install` — they are alternative install modes.

---

## Phase 5 — `docker/pipx/Dockerfile` (Method E)

**What to implement (copy-based):**
1. Base image: `python:3.11-slim-bookworm` (same as Phases 2–4).
2. apt block + GDAL ENV vars: copy from Phase 2.
3. Install pipx and SYMFLUENCE:
   ```
   RUN pip install --no-cache-dir pipx \
       && pipx ensurepath
   ENV PATH=/root/.local/bin:$PATH
   RUN pipx install symfluence
   RUN symfluence binary install
   ```
   The verbatim user-facing command (README.md line 48) is `pipx install symfluence`; the rest is
   plumbing required because pipx is not preinstalled on the slim base.
4. ENTRYPOINT/CMD: copy from Phase 2.

**Documentation references:**
- README.md line 48 (`pipx install symfluence`).
- pipx upstream docs for `ensurepath` (referenced by name only — do not copy URL).

**Verification checklist:**
- `docker build -f docker/pipx/Dockerfile -t symfluence:pipx .` succeeds.
- `docker run --rm symfluence:pipx --help` prints SYMFLUENCE help.
- `docker run --rm --entrypoint pipx symfluence:pipx list` shows `symfluence`.

**Anti-pattern guards:**
- Do not install pipx via `apt-get install pipx` on Debian 12 — versions are stale and may
  conflict with the Python 3.11 interpreter.
- Do not skip `pipx ensurepath` + `ENV PATH` — pipx installs into a venv under `~/.local`.

---

## Phase 6 — `docker/npm/Dockerfile` (Method A)

**What to implement (copy-based):**
1. Base image: `node:20-bookworm-slim` (matches the Debian 12 platform that npm/package.json
   targets for Linux x86_64; Apple Silicon is not buildable from a typical CI Linux host).
2. Install the **runtime-only** apt list verbatim from `docs/SYSTEM_REQUIREMENTS.md` lines 34–44
   (libnetcdf19, libnetcdff7, libhdf5-103, libgdal32, libproj25, libgeos3.11.1, libopenmpi3)
   plus `python3 ca-certificates`. Do NOT install build toolchain — npm ships pre-built binaries.
3. Install with the verbatim README.md line 75 command:
   ```
   RUN npm install -g symfluence
   ```
4. Sanity-check the binaries with the verbatim README.md lines 78–82 commands:
   ```
   RUN symfluence binary info \
       && symfluence binary doctor
   ```
5. ENTRYPOINT/CMD: copy from Phase 2.

**Documentation references:**
- README.md lines 75, 78–82 (`npm install -g symfluence`, `binary info`, `binary doctor`).
- `tools/npm/README.md` lines 9–13 (list of bundled binaries — do not paraphrase).
- `docs/SYSTEM_REQUIREMENTS.md` lines 34–44 (runtime apt list).
- `npm/package.json` (declared platform support).

**Verification checklist:**
- `docker build --platform=linux/amd64 -f docker/npm/Dockerfile -t symfluence:npm .` succeeds.
- `docker run --rm symfluence:npm binary info` lists the bundled binaries (SUMMA, mizuRoute,
  FUSE, NGEN, TauDEM at minimum).
- `docker run --rm symfluence:npm --help` prints SYMFLUENCE help.
- The image does **not** contain `gcc`, `gfortran`, `cmake` — verify with
  `docker run --rm --entrypoint sh symfluence:npm -c 'command -v gfortran || echo absent'`.

**Anti-pattern guards:**
- Do not run `symfluence binary install` here — it would compile from source and defeat the npm
  pre-built distribution path.
- Do not use a Python base image — npm is the entry point, and the npm package handles its own
  Python interop.
- Do not target ARM64 from a non-ARM host without `--platform=linux/arm64` and matching upstream
  pre-built binaries.

---

## Phase 7 — `docker/conda/Dockerfile` (Method F)

**What to implement (copy-based):**
1. Base image: `condaforge/miniforge3:24.11.3-2` — copy from existing root `Dockerfile` line 1.
2. Recreate the verbatim conda command sequence from installation.html "Windows" subsection:
   ```
   RUN conda create -y -n symfluence python=3.11 \
       && conda install -y -n symfluence -c conda-forge \
            gdal geopandas rasterio netcdf4 hdf5
   ```
3. Activate environment for subsequent `RUN` layers:
   ```
   ENV PATH=/opt/conda/envs/symfluence/bin:$PATH
   ENV CONDA_DEFAULT_ENV=symfluence
   ```
4. Install symfluence:
   ```
   RUN pip install --no-cache-dir symfluence
   RUN symfluence binary install
   ```
5. ENTRYPOINT/CMD: copy from existing root `Dockerfile`.

**Documentation references:**
- installation.html "Windows" subsection (verbatim conda command block — also reproduced in
  Phase 0 above).
- Existing root `Dockerfile` (base image choice + multi-stage cues — but Phase 7 is **single
  stage** unless a clear size win justifies two; multi-stage is Phase 8 polish).

**Verification checklist:**
- `docker build -f docker/conda/Dockerfile -t symfluence:conda .` succeeds.
- `docker run --rm symfluence:conda --help` prints SYMFLUENCE help.
- `docker run --rm --entrypoint conda symfluence:conda list -n symfluence | grep -E 'gdal|rasterio'`
  shows both packages.

**Anti-pattern guards:**
- Do not use `conda activate` inside `RUN` — `RUN` invokes `/bin/sh -c`, which does not source
  conda init scripts. Use the `ENV PATH=...` pattern (Phase 7 step 3).
- Do not install GDAL via apt in this image — conda-forge GDAL is the documented choice for this
  method and mixing them silently breaks at runtime.

---

## Phase 8 — `docker/source/Dockerfile` (Method G — bootstrap from clone)

**What to implement (copy-based):**
1. Base image: `python:3.11-slim-bookworm` (matches `installation.rst` Manual Setup, which uses
   `python3 -m venv venv`).
2. Build-toolchain apt block + GDAL ENV vars: copy from Phase 2.
3. Add `git` to the apt list (it's already in Phase 2's list above).
4. Implement the verbatim installation.html "Development Installation" sequence:
   ```
   WORKDIR /opt
   RUN git clone https://github.com/symfluence-org/SYMFLUENCE.git
   WORKDIR /opt/SYMFLUENCE
   RUN ./scripts/symfluence-bootstrap --install
   ENV PATH=/opt/SYMFLUENCE/venv/bin:$PATH
   ```
5. ENTRYPOINT/CMD: copy from Phase 2.

**Variant** (manual fallback per installation.html "Manual Setup"): build a second target inside
the same Dockerfile (`AS manual`) that does:
   ```
   FROM <base> AS manual
   COPY . /opt/SYMFLUENCE
   WORKDIR /opt/SYMFLUENCE
   RUN python3 -m venv venv \
       && ./venv/bin/pip install -e .
   ENV PATH=/opt/SYMFLUENCE/venv/bin:$PATH
   ```
Build with `--target manual` to get the editable-install variant.

**Documentation references:**
- installation.html "Development Installation" + "Manual Setup".
- `installation.rst` lines 67–72 (manual venv).
- README.md lines 60–64 (clone + bootstrap).

**Verification checklist:**
- `docker build -f docker/source/Dockerfile -t symfluence:source .` succeeds.
- `docker build --target manual -f docker/source/Dockerfile -t symfluence:source-manual .`
  succeeds.
- `docker run --rm symfluence:source --help` prints SYMFLUENCE help.
- `docker run --rm --entrypoint sh symfluence:source-manual -c 'pip show symfluence | grep Location'`
  shows the path is `/opt/SYMFLUENCE` (proves editable install).

**Anti-pattern guards:**
- Do not pass `--upgrade` or any flag to `symfluence-bootstrap` other than `--install` — no other
  flag is documented on readthedocs.
- Do not skip the `ENV PATH=.../venv/bin:$PATH` line — the bootstrap script creates the venv but
  does not provide a system-level `symfluence` shim.
- Do not `git clone` into `/` or another path — `WORKDIR /opt/SYMFLUENCE` is what the bootstrap
  script expects to find.

---

## Phase 9 — `docker-compose.yaml` updates + README docs

**What to implement (copy-based):**
1. Decide whether to extend `docker-compose.yaml` with one service per method, or document the
   `docker build -f docker/<method>/Dockerfile` invocations in `README.md`. Default
   recommendation: **README docs only** for now (keep compose lean; add compose services later if
   user asks).
2. Add a new section to `README.md` titled "Containerised installs" listing each method with its
   `docker build` and `docker run` command. Use the exact image-tag convention from Phase 1.
3. Do NOT modify the existing root `Dockerfile`.

**Documentation references:**
- `docker-compose.yaml` (existing service shape).
- README.md Installation section header style.

**Verification checklist:**
- `git diff README.md` shows only the new "Containerised installs" section.
- `git diff Dockerfile docker-compose.yaml` shows no changes (unless the user explicitly asks).
- All seven `docker build` commands in the new README section copy-paste cleanly.

**Anti-pattern guards:**
- Do not duplicate install commands inside Dockerfiles into the README — link to `docker/<method>/`
  instead.
- Do not mark any new section "deprecated" or "experimental" without user confirmation.

---

## Final Phase — Verification (cross-cutting)

**Verification subagent / runner checklist:**
1. `find docker -name Dockerfile | sort` → exactly seven results.
2. For each of the seven, run `docker build -f <path> -t symfluence:<method> .` from the repo
   root. **Note:** these builds are heavy; expect ≥10 min for any method that triggers
   `symfluence binary install`. Allow `npm` and the runtime-only paths to short-circuit faster.
3. For each successful image, run `docker run --rm symfluence:<method> --help` and confirm
   non-empty output.
4. Anti-pattern grep on the new files:
   ```
   grep -rE 'symfluence install-binaries|symfluence install\b|conda activate' docker/
   ```
   → must be empty.
5. Anti-pattern grep for invented extras:
   ```
   grep -rE 'symfluence\[(?!gdal|baseflow|r|dev|test|docs|notebook|gui|tui|alos|hpc|all)' docker/
   ```
   → must be empty.
6. Confirm `.dockerignore` still excludes Python/build artefacts; no edits needed unless a build
   above failed because of a context size or include error.
7. `pytest tests/unit/ -v --tb=short` (per repo `.claude/rules/issue-fix.md`) — should pass; this
   plan touches no Python code, so the pre-existing unit-test baseline must remain green.
8. Stage and review with `git status` + `git diff --stat docker/ README.md` before committing.
   Conventional Commits per project rules: `feat(docker): add per-method install Dockerfiles`.

**Exit criteria:**
- Seven Dockerfiles exist under `docker/`.
- Each builds and produces a working `symfluence --help` (npm and pre-built paths verified
  without invoking `binary install`).
- README has a new "Containerised installs" section.
- No edits to root `Dockerfile`, `docker-compose.yaml`, or any Python source.
