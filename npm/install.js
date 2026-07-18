#!/usr/bin/env node
/**
 * SYMFLUENCE npm installer
 * Downloads and extracts pre-built binaries from GitHub Releases
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');
const { execSync } = require('child_process');
const { getPlatform, getPlatformName } = require('./lib/platform');

const PACKAGE_VERSION = require('./package.json').version;
const GITHUB_REPO = 'symfluence-org/SYMFLUENCE';

/**
 * Construct the download URL for the current platform
 * @param {string} platform - Platform identifier (e.g., 'macos-arm64')
 * @returns {string} Full download URL
 */
function getDownloadUrl(platform) {
  const tag = `v${PACKAGE_VERSION}`;
  const filename = `symfluence-tools-${tag}-${platform}.tar.gz`;
  return `https://github.com/${GITHUB_REPO}/releases/download/${tag}/${filename}`;
}

/**
 * Download a file from URL with progress tracking
 * @param {string} url - URL to download from
 * @param {string} dest - Destination file path
 * @returns {Promise<void>}
 */
async function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);

    const request = https.get(url, {
      headers: { 'User-Agent': 'symfluence-npm-installer' }
    }, (response) => {
      // Handle redirects
      if (response.statusCode === 302 || response.statusCode === 301) {
        file.close();
        fs.unlinkSync(dest);
        downloadFile(response.headers.location, dest).then(resolve).catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(dest);
        reject(new Error(
          `Download failed: ${response.statusCode} ${response.statusMessage}\n` +
          `URL: ${url}`
        ));
        return;
      }

      const totalBytes = parseInt(response.headers['content-length'], 10);
      let downloadedBytes = 0;
      let lastPercent = -1;

      response.on('data', (chunk) => {
        downloadedBytes += chunk.length;
        const percent = Math.floor((downloadedBytes / totalBytes) * 100);

        // Only update display every 5% to reduce output noise
        if (percent !== lastPercent && percent % 5 === 0) {
          const mb = (downloadedBytes / 1024 / 1024).toFixed(1);
          const totalMb = (totalBytes / 1024 / 1024).toFixed(1);
          process.stdout.write(`\r📥 Downloading... ${percent}% (${mb}/${totalMb} MB)`);
          lastPercent = percent;
        }
      });

      response.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log('\n✅ Download complete');
        resolve();
      });

      file.on('error', (err) => {
        fs.unlinkSync(dest);
        reject(err);
      });
    });

    request.on('error', (err) => {
      if (fs.existsSync(dest)) {
        fs.unlinkSync(dest);
      }
      reject(err);
    });
  });
}

/**
 * Verify file checksum against published SHA256
 * @param {string} file - Path to file to verify
 * @param {string} checksumUrl - URL of .sha256 file
 * @returns {Promise<void>}
 */
async function verifyChecksum(file, checksumUrl) {
  console.log('🔐 Verifying checksum...');

  try {
    // Download checksum file
    const checksumData = await new Promise((resolve, reject) => {
      let data = '';
      https.get(checksumUrl, {
        headers: { 'User-Agent': 'symfluence-npm-installer' }
      }, (res) => {
        if (res.statusCode === 302 || res.statusCode === 301) {
          // Follow redirect
          https.get(res.headers.location, (redirectRes) => {
            redirectRes.on('data', chunk => data += chunk);
            redirectRes.on('end', () => resolve(data));
          }).on('error', reject);
          return;
        }
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve(data));
      }).on('error', reject);
    });

    // Extract expected hash (format: "hash  filename")
    const expectedHash = checksumData.trim().split(/\s+/)[0];

    // Calculate actual hash
    const fileBuffer = fs.readFileSync(file);
    const hash = crypto.createHash('sha256');
    hash.update(fileBuffer);
    const actualHash = hash.digest('hex');

    if (expectedHash.toLowerCase() !== actualHash.toLowerCase()) {
      // A real mismatch means a corrupted or tampered download — never proceed.
      throw new Error(
        'Checksum mismatch! File may be corrupted or tampered with.\n' +
        `  Expected: ${expectedHash}\n` +
        `  Actual:   ${actualHash}\n` +
        '  Delete the download and re-run the install.'
      );
    }

    console.log('✅ Checksum verified');
  } catch (err) {
    if (err.message.startsWith('Checksum mismatch')) {
      throw err;
    }
    // Only tolerate failure to *fetch* the checksum file (offline mirror, etc.)
    console.warn('⚠️  Could not verify checksum:', err.message);
    console.warn('   Proceeding anyway, but installation may be corrupted...');
  }
}

/**
 * Extract tarball to destination directory
 * @param {string} tarball - Path to tarball
 * @param {string} destDir - Destination directory
 */
function extractTarball(tarball, destDir) {
  console.log('📦 Extracting binaries...');

  // On Windows (MSYS/MinGW), tar interprets D: as a remote host.
  // --force-local fixes this but is not supported by BSD tar (macOS).
  // Also convert backslash paths to forward slashes — MSYS tar cannot
  // handle Windows-style backslash paths in -C arguments.
  const forceLocal = process.platform === 'win32' ? '--force-local ' : '';
  const tarPath = process.platform === 'win32' ? tarball.replace(/\\/g, '/') : tarball;
  const destPath = process.platform === 'win32' ? destDir.replace(/\\/g, '/') : destDir;
  const extractCmd = `tar ${forceLocal}-xzf "${tarPath}" -C "${destPath}" --strip-components=1`;

  try {
    execSync(extractCmd, { stdio: 'inherit' });
    console.log('✅ Extraction complete');
  } catch (err) {
    throw new Error(`Extraction failed: ${err.message}`);
  }
}

/**
 * Check whether a shell command exists / succeeds silently.
 * @param {string} cmd - Command to run
 * @returns {boolean}
 */
function commandExists(cmd) {
  try {
    execSync(cmd, { stdio: 'ignore', timeout: 10000 });
    return true;
  } catch {
    return false;
  }
}

/**
 * Verify the external tooling install.js itself shells out to is present.
 *
 * Node provides https/crypto for download + checksum, but extraction relies on
 * the system `tar`. A missing `tar` previously surfaced as an opaque
 * "Extraction failed" only after a multi-hundred-MB download. Fail fast with an
 * actionable message instead. (#156 G6 — undeclared install-time tooling)
 *
 * @throws {Error} if a hard-required tool is missing
 */
function assertInstallTooling() {
  const tarCmd = process.platform === 'win32' ? 'tar --version' : 'tar --version';
  if (!commandExists(tarCmd)) {
    throw new Error(
      "'tar' was not found on PATH.\n" +
      '   tar is required to extract the pre-built binary tarball.\n' +
      (process.platform === 'win32'
        ? '   Windows 10+ ships bsdtar; ensure C:\\Windows\\System32 is on PATH.\n'
        : '   Install it via your package manager (e.g. apt-get install tar).\n')
    );
  }
}

/**
 * Parse a "major.minor" version string into a comparable [major, minor] tuple.
 * @returns {number[]|null}
 */
function parseVersionPair(text) {
  const m = String(text).match(/(\d+)\.(\d+)/);
  return m ? [parseInt(m[1], 10), parseInt(m[2], 10)] : null;
}

/** Compare two [major, minor] tuples. Returns <0, 0, or >0. */
function cmpVersion(a, b) {
  return a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1];
}

/**
 * Best-effort host glibc version (Linux only).
 * @returns {number[]|null} [major, minor] or null
 */
function hostGlibcVersion() {
  for (const cmd of ['getconf GNU_LIBC_VERSION 2>/dev/null', 'ldd --version 2>/dev/null']) {
    try {
      const out = execSync(cmd, { encoding: 'utf8', timeout: 5000 }).split('\n')[0];
      const v = parseVersionPair(out);
      if (v) return v;
    } catch { /* try next */ }
  }
  return null;
}

/**
 * Max GLIBC_x.y symbol version required across one or more ELF files.
 *
 * Scans every supplied file — the real baseline is driven by the bundled shared
 * libraries (e.g. libgfortran.so.5), not just the main executable, so callers
 * must pass dist/lib/*.so too or the result understates the true requirement.
 *
 * @param {string[]} files - ELF binaries/libraries to inspect
 * @returns {number[]|null} max [major, minor] or null if undeterminable
 */
function requiredGlibcVersion(files) {
  let max = null;
  for (const f of files) {
    for (const tool of ['objdump -T', 'readelf -V']) {
      try {
        const out = execSync(
          `${tool} "${f}" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\\.[0-9]+' | sort -V | tail -1`,
          { encoding: 'utf8', timeout: 10000 }
        ).trim();
        const v = parseVersionPair(out.replace('GLIBC_', ''));
        if (v) {
          if (max === null || cmpVersion(v, max) > 0) max = v;
          break; // got a value from this tool; move to next file
        }
      } catch { /* try next tool */ }
    }
  }
  return max;
}

/**
 * Shared libraries a binary cannot resolve, honouring the bundled dist/lib
 * (which the runtime wrapper also prepends to LD_LIBRARY_PATH). Linux only.
 * @returns {string[]} unresolved "lib... => not found" lines
 */
function unresolvedLibraries(binary, distLibDir) {
  if (process.platform !== 'linux') return [];
  const env = {
    ...process.env,
    LD_LIBRARY_PATH: `${distLibDir}${path.delimiter}${process.env.LD_LIBRARY_PATH || ''}`,
  };
  let out = '';
  try {
    out = execSync(`ldd "${binary}" 2>&1`, { encoding: 'utf8', timeout: 15000, env });
  } catch (err) {
    out = (err.stdout || '').toString() + (err.stderr || '').toString();
  }
  return out
    .split('\n')
    .filter((line) => /not found/i.test(line))
    .map((line) => line.trim());
}

/**
 * Post-extract sanity check of the bundled native binaries.
 *
 * Surfaces the two distro-portability failure modes G6 calls out — a libnetcdff
 * (or HDF5) soname the host doesn't provide, and a host glibc older than the
 * binaries were built against — at install time with actionable guidance,
 * instead of as an opaque loader error on first model run. Non-fatal: the npm
 * shim's built-in commands still work, and the system-dep install may yet
 * provide the missing libraries.
 */
function verifyBundledBinaries(distDir) {
  if (process.platform === 'win32') return; // libs bundled in the tarball

  const binDir = path.join(distDir, 'bin');
  const libDir = path.join(distDir, 'lib');
  if (!fs.existsSync(binDir)) return;

  // Prefer the Fortran models that link libnetcdff/HDF5 — they exercise the
  // soname-variance path. Fall back to whatever is present.
  const preferred = ['summa', 'mizuroute', 'fuse'];
  const present = preferred
    .map((n) => path.join(binDir, n))
    .filter((p) => fs.existsSync(p));
  if (present.length === 0) return;

  console.log('\n🔎 Verifying bundled binaries link cleanly on this host...');

  const missing = new Set();
  for (const bin of present) {
    for (const line of unresolvedLibraries(bin, libDir)) {
      // keep just the soname (e.g. "libnetcdff.so.7")
      missing.add(line.split('=>')[0].trim() || line);
    }
  }

  if (missing.size > 0) {
    console.warn('\n⚠️  Some bundled binaries are missing shared libraries on this system:');
    for (const lib of missing) console.warn(`     - ${lib}`);
    console.warn(
      '\n   This is usually a libnetcdff/HDF5 soname that differs across distros.\n' +
      '   Fixes (any one):\n' +
      '     • Install the matching dev packages (e.g. apt-get install libnetcdff-dev libhdf5-dev),\n' +
      '     • or build from source: symfluence binary install\n'
    );
  }

  // glibc baseline (Linux only) — read what the binary AND its bundled shared
  // libraries actually require (libgfortran etc. usually drive the baseline).
  if (process.platform === 'linux') {
    const host = hostGlibcVersion();
    let elfFiles = [present[0]];
    try {
      if (fs.existsSync(libDir)) {
        elfFiles = elfFiles.concat(
          fs.readdirSync(libDir)
            .filter((f) => f.includes('.so'))
            .map((f) => path.join(libDir, f))
        );
      }
    } catch { /* best-effort */ }
    const required = requiredGlibcVersion(elfFiles);
    if (host && required && cmpVersion(host, required) < 0) {
      console.warn(
        `\n⚠️  Host glibc ${host.join('.')} is older than the binaries' baseline ` +
        `(glibc ${required.join('.')}).\n` +
        '   The pre-built Linux binaries will fail to load. Use a newer distro\n' +
        '   or build from source: symfluence binary install\n'
      );
    } else if (host && required) {
      console.log(`   glibc ${host.join('.')} ≥ required ${required.join('.')} ✓`);
    }
  }

  if (missing.size === 0) {
    console.log('   ✅ Bundled binaries resolve their libraries.');
  }
}

/**
 * Runtime C/Fortran libraries required by SYMFLUENCE models (SUMMA, FUSE, etc.)
 * Each entry lists one or more detection commands and per-package-manager names.
 */
const RUNTIME_DEPS = [
  {
    name: 'NetCDF-C',
    detect: ['nc-config --version'],
    brew: 'netcdf',
    apt: 'libnetcdf-dev',
    dnf: 'netcdf',
    conda: 'netcdf4',
  },
  {
    name: 'NetCDF-Fortran',
    detect: ['nf-config --version'],
    brew: 'netcdf-fortran',
    apt: 'libnetcdff-dev',
    dnf: 'netcdf-fortran',
    conda: 'netcdf-fortran',
  },
  {
    name: 'HDF5',
    detect: ['h5cc -showconfig', 'h5dump --version'],
    brew: 'hdf5',
    apt: 'libhdf5-dev',
    dnf: 'hdf5',
    conda: 'hdf5',
  },
  {
    name: 'GDAL',
    detect: ['gdal-config --version', 'gdalinfo --version'],
    brew: 'gdal',
    apt: 'gdal-bin libgdal-dev',
    dnf: 'gdal',
    conda: 'gdal',
  },
  {
    name: 'OpenBLAS',
    detect: ['dpkg -s libopenblas0 2>/dev/null | grep "Status: install ok"',
             'ldconfig -p 2>/dev/null | grep libopenblas'],
    brew: 'openblas',
    apt: 'libopenblas0',
    dnf: 'openblas',
    conda: 'openblas',
  },
];

/**
 * Check whether a single runtime dependency is available.
 * @param {{ name: string, detect: string[] }} dep
 * @returns {boolean}
 */
function checkDep(dep) {
  return dep.detect.some(cmd => commandExists(cmd));
}

/**
 * Detect the best available system package manager.
 * @returns {{ name: string, installCmd: string, key: string } | null}
 */
function detectPackageManager() {
  // Conda (cross-platform, checked first)
  if (process.env.CONDA_PREFIX) {
    return { name: 'conda', installCmd: 'conda install -y -c conda-forge', key: 'conda' };
  }

  if (process.platform === 'darwin') {
    if (commandExists('brew --version')) {
      return { name: 'Homebrew', installCmd: 'brew install', key: 'brew' };
    }
    return null;
  }

  if (process.platform === 'linux') {
    if (commandExists('apt-get --version')) {
      return { name: 'apt', installCmd: 'sudo apt-get install -y', key: 'apt' };
    }
    if (commandExists('dnf --version')) {
      return { name: 'dnf', installCmd: 'sudo dnf install -y', key: 'dnf' };
    }
  }

  return null;
}

/**
 * Print manual installation instructions for missing dependencies.
 * @param {{ name: string }[]} missing
 */
function printManualInstructions(missing) {
  const names = missing.map(d => d.name).join(', ');
  console.warn(`\n⚠️  Could not auto-install system dependencies: ${names}`);
  console.warn('   Please install them manually:\n');

  if (process.platform === 'darwin') {
    const pkgs = missing.map(d => d.brew).join(' ');
    console.warn(`   # macOS (Homebrew)`);
    console.warn(`   brew install ${pkgs}`);
    console.warn(`   # Install Homebrew: https://brew.sh\n`);
  } else if (process.platform === 'linux') {
    const aptPkgs = missing.map(d => d.apt).join(' ');
    const dnfPkgs = missing.map(d => d.dnf).join(' ');
    console.warn(`   # Debian/Ubuntu`);
    console.warn(`   sudo apt-get install -y ${aptPkgs}\n`);
    console.warn(`   # Fedora/RHEL`);
    console.warn(`   sudo dnf install -y ${dnfPkgs}\n`);
  }

  const condaPkgs = missing.map(d => d.conda).join(' ');
  console.warn(`   # Conda (any platform)`);
  console.warn(`   conda install -c conda-forge ${condaPkgs}\n`);
  console.warn('   📖 https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md\n');
}

/**
 * Detect and auto-install system-level runtime dependencies (NetCDF, HDF5, GDAL).
 * Non-fatal: prints manual instructions on failure.
 */
function tryInstallSystemDeps() {
  // Skip on Windows (libraries are bundled in the tarball)
  if (process.platform === 'win32') {
    return;
  }

  // Allow users to opt out
  if (process.env.SYMFLUENCE_SKIP_SYSTEM_DEPS === '1') {
    console.log('\n📦 Skipping system dependency check (SYMFLUENCE_SKIP_SYSTEM_DEPS=1)\n');
    return;
  }

  console.log('\n🔍 Checking system dependencies...\n');

  const found = [];
  const missing = [];

  for (const dep of RUNTIME_DEPS) {
    if (checkDep(dep)) {
      console.log(`   ✅ ${dep.name}`);
      found.push(dep);
    } else {
      console.log(`   ❌ ${dep.name} — not found`);
      missing.push(dep);
    }
  }

  if (missing.length === 0) {
    console.log('\n   All system dependencies found.\n');
    return;
  }

  console.log(`\n   ${missing.length} missing dependenc${missing.length === 1 ? 'y' : 'ies'}, attempting auto-install...\n`);

  const pm = detectPackageManager();
  if (!pm) {
    printManualInstructions(missing);
    return;
  }

  // Build install command from the per-manager package names
  const pkgs = missing.map(d => d[pm.key]).join(' ');
  const cmd = `${pm.installCmd} ${pkgs}`;

  console.log(`   Using ${pm.name}: ${cmd}\n`);

  try {
    execSync(cmd, { stdio: 'inherit', timeout: 300000 });
    console.log(`\n✅ System dependencies installed via ${pm.name}`);
  } catch (err) {
    console.warn(`\n⚠️  ${pm.name} install failed: ${err.message}`);
    printManualInstructions(missing);
  }
}

/**
 * Install the SYMFLUENCE Python package.
 *
 * Tries uv, pip3, pip in order. Fails the postinstall (exit 1) if none
 * succeed — the npm package is a wrapper around the Python CLI, so a missing
 * Python install means everything except the built-in commands is broken,
 * and silently warning at install time produces a confusing error at first
 * forwarded-command run.
 *
 * Set SYMFLUENCE_OPTIONAL_PYTHON=1 to opt out (e.g. when only the built-in
 * commands `version`, `path`, `help`, `info` are needed).
 */
function tryInstallPython() {
  console.log('\n🐍 Installing SYMFLUENCE Python package...\n');

  // Pinned first so the Python package matches this npm release; fall back to
  // latest if the pinned version is not yet on PyPI (npm and PyPI publish from
  // the same tag but land asynchronously).
  const specs = [`symfluence==${PACKAGE_VERSION}`, 'symfluence'];

  const strategies = [
    { check: 'uv --version', install: 'uv pip install --upgrade', label: 'uv' },
    { check: 'pip3 --version', install: 'pip3 install --upgrade', label: 'pip3' },
    { check: 'pip --version', install: 'pip install --upgrade', label: 'pip' },
  ];

  for (const { check, install, label } of strategies) {
    try {
      execSync(check, { stdio: 'ignore', timeout: 10000 });
    } catch {
      continue; // tool not available
    }

    for (const spec of specs) {
      try {
        console.log(`   Using ${label} (${spec})...`);
        // 10 min: heavy scientific stack (torch, geopandas, etc.) can exceed
        // the previous 120 s budget on slow networks or under emulation.
        execSync(`${install} "${spec}"`, { stdio: 'inherit', timeout: 600000 });
        console.log(`\n✅ Python package installed via ${label}`);
        return;
      } catch (err) {
        console.warn(`\n⚠️  ${label} install of ${spec} failed: ${err.message}`);
        // try next spec, then next strategy
      }
    }
  }

  const message =
    '\n❌ Could not auto-install the SYMFLUENCE Python package.\n\n' +
    '   The npm package is a wrapper around the Python CLI; without it,\n' +
    '   only the built-in commands (version, path, help, info) will work.\n\n' +
    '   Install Python 3 and one of: uv, pip3, pip — then re-run:\n' +
    '     npm install -g symfluence\n\n' +
    '   To install the npm shim alone (built-ins only), set:\n' +
    '     SYMFLUENCE_OPTIONAL_PYTHON=1\n';

  if (process.env.SYMFLUENCE_OPTIONAL_PYTHON === '1') {
    console.warn(message);
    return;
  }

  console.error(message);
  process.exit(1);
}

/**
 * Post-install check: confirm the Python CLI resolves and report whether its
 * version matches this npm release. Non-fatal — the unpinned fallback can
 * legitimately install a different version while a tag's PyPI publish is
 * still landing, and the runtime wrapper warns on every skewed invocation.
 */
function verifyPythonVersion(distDir) {
  const candidates = ['python3', 'python'];

  for (const py of candidates) {
    let out;
    try {
      // stdio array keeps probe stderr (tracebacks, import warnings) out of the install log
      out = execSync(`${py} -m symfluence --version`,
        { encoding: 'utf8', timeout: 15000, stdio: ['ignore', 'pipe', 'ignore'] });
    } catch {
      continue; // try next interpreter
    }
    const m = out.match(/(\d+\.\d+\.\d+)/);
    const version = m ? m[1] : null;
    if (version === PACKAGE_VERSION) {
      console.log(`\n✅ Python package ${version} matches the npm package`);
    } else {
      console.warn(
        `\n⚠️  Python package reports ${version || 'an unknown version'}, npm package is ${PACKAGE_VERSION}.\n` +
        `   Sync manually with: pip install --upgrade "symfluence==${PACKAGE_VERSION}"`
      );
    }
    return;
  }
  console.warn('\n⚠️  Could not verify the installed Python package version.');
}

/**
 * Main installation function
 */
async function install() {
  console.log('╔════════════════════════════════════════════╗');
  console.log('║   SYMFLUENCE Binary Installer              ║');
  console.log('╚════════════════════════════════════════════╝\n');

  // Detect platform
  let platform;
  try {
    platform = getPlatform();
  } catch (err) {
    console.error('❌', err.message);
    console.error('\n📖 For manual installation, see:');
    console.error('   https://github.com/symfluence-org/SYMFLUENCE#installation\n');
    process.exit(1);
  }

  console.log(`📍 Platform: ${getPlatformName()} (${platform})`);
  console.log(`📦 Version: ${PACKAGE_VERSION}\n`);

  // Fail fast on missing install-time tooling before a large download.
  try {
    assertInstallTooling();
  } catch (err) {
    console.error('❌', err.message);
    process.exit(1);
  }

  const url = getDownloadUrl(platform);
  const checksumUrl = `${url}.sha256`;

  console.log(`🔗 Downloading from GitHub Releases...`);
  console.log(`   ${url}\n`);

  const distDir = path.join(__dirname, 'dist');
  const tarballPath = path.join(__dirname, 'symfluence-tools.tar.gz');

  // Clean and create dist directory
  if (fs.existsSync(distDir)) {
    console.log('🧹 Cleaning previous installation...');
    fs.rmSync(distDir, { recursive: true, force: true });
  }
  fs.mkdirSync(distDir, { recursive: true });

  try {
    // Download tarball
    await downloadFile(url, tarballPath);

    // Verify checksum
    await verifyChecksum(tarballPath, checksumUrl);

    // Extract
    extractTarball(tarballPath, distDir);

    // Cleanup tarball
    fs.unlinkSync(tarballPath);

    // Surface distro-portability issues (soname / glibc) now, not at first run.
    verifyBundledBinaries(distDir);

    // Install system libraries and the Python package (uv/pip3/pip)
    tryInstallSystemDeps();
    tryInstallPython();

    if (process.env.SYMFLUENCE_OPTIONAL_PYTHON !== '1') {
      verifyPythonVersion(distDir);
    }

    // Display installation info
    console.log('\n╔════════════════════════════════════════════╗');
    console.log('║   🎉 Installation Complete!                ║');
    console.log('╚════════════════════════════════════════════╝\n');

    console.log('📦 Installed Tools:');
    const binDir = path.join(distDir, 'bin');
    if (fs.existsSync(binDir)) {
      const tools = fs.readdirSync(binDir).filter(f => {
        const fullPath = path.join(binDir, f);
        return fs.statSync(fullPath).isFile();
      });
      tools.forEach(tool => console.log(`   ✓ ${tool}`));

      // MPI runtime detection
      if (fs.existsSync(path.join(binDir, 'mpirun'))) {
        console.log('\n   MPI runtime bundled (no separate MPI install needed)');
      }
    }

    console.log('\n📖 Next Steps:');
    console.log('   1. Check installation: symfluence --help');
    console.log('   2. Run a bundled binary: symfluence binary summa --version');
    console.log('   3. View available tools: ls $(npm root -g)/symfluence/dist/bin\n');

    console.log('📚 Documentation: https://github.com/symfluence-org/SYMFLUENCE\n');

  } catch (err) {
    console.error('\n❌ Installation failed:', err.message);
    console.error('\n📖 Troubleshooting:');
    console.error('   1. Check your internet connection');
    console.error('   2. Verify the release exists:');
    console.error(`      https://github.com/${GITHUB_REPO}/releases/tag/v${PACKAGE_VERSION}`);
    console.error('   3. Check system requirements:');
    console.error('      https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md');
    console.error('   4. Try manual installation:');
    console.error('      https://github.com/symfluence-org/SYMFLUENCE#installation\n');

    // Clean up on failure
    if (fs.existsSync(tarballPath)) {
      fs.unlinkSync(tarballPath);
    }
    if (fs.existsSync(distDir)) {
      fs.rmSync(distDir, { recursive: true, force: true });
    }

    process.exit(1);
  }
}

// Run installer if executed directly (not required)
if (require.main === module) {
  install();
}

// Exported for testing / reuse (e.g. the multi-distro CI matrix). Requiring
// this module does not run install() — that only happens when run directly.
module.exports = {
  assertInstallTooling,
  parseVersionPair,
  cmpVersion,
  hostGlibcVersion,
  requiredGlibcVersion,
  unresolvedLibraries,
  verifyBundledBinaries,
  verifyPythonVersion,
};
