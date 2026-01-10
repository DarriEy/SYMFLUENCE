/**
 * SYMFLUENCE npm package - Programmatic API
 * Provides access to binary paths and metadata
 */

const fs = require('fs');
const path = require('path');
const { getPlatform, getPlatformName, isPlatformSupported } = require('./lib/platform');

const DIST_DIR = path.join(__dirname, 'dist');
const BIN_DIR = path.join(DIST_DIR, 'bin');
const TOOLCHAIN_FILE = path.join(DIST_DIR, 'toolchain.json');

/**
 * Check if binaries are installed
 * @returns {boolean}
 */
function isInstalled() {
  return fs.existsSync(DIST_DIR) && fs.existsSync(BIN_DIR);
}

/**
 * Get the binary directory path
 * @returns {string|null} Path to bin directory, or null if not installed
 */
function getBinDir() {
  return isInstalled() ? BIN_DIR : null;
}

/**
 * Get path to a specific tool
 * @param {string} toolName - Name of the tool (e.g., 'summa', 'mizuroute')
 * @returns {string|null} Full path to tool, or null if not found
 */
function getToolPath(toolName) {
  if (!isInstalled()) {
    return null;
  }

  const toolPath = path.join(BIN_DIR, toolName);
  return fs.existsSync(toolPath) ? toolPath : null;
}

/**
 * Get list of installed tools
 * @returns {Array<string>} Array of tool names
 */
function getInstalledTools() {
  if (!isInstalled()) {
    return [];
  }

  try {
    return fs.readdirSync(BIN_DIR)
      .filter(f => {
        const fullPath = path.join(BIN_DIR, f);
        const stats = fs.statSync(fullPath);
        return stats.isFile() && (stats.mode & 0o111); // Executable
      })
      .sort();
  } catch (err) {
    return [];
  }
}

/**
 * Get toolchain metadata
 * @returns {Object|null} Toolchain metadata, or null if not available
 */
function getToolchain() {
  if (!fs.existsSync(TOOLCHAIN_FILE)) {
    return null;
  }

  try {
    return JSON.parse(fs.readFileSync(TOOLCHAIN_FILE, 'utf8'));
  } catch (err) {
    return null;
  }
}

/**
 * Get package version
 * @returns {string}
 */
function getVersion() {
  return require('./package.json').version;
}

/**
 * Get all available paths for environment setup
 * @returns {Object} Object with paths
 */
function getPaths() {
  return {
    dist: DIST_DIR,
    bin: BIN_DIR,
    toolchain: TOOLCHAIN_FILE,
  };
}

module.exports = {
  // Installation checks
  isInstalled,
  isPlatformSupported,

  // Paths
  getBinDir,
  getToolPath,
  getPaths,

  // Tool information
  getInstalledTools,
  getToolchain,
  getVersion,

  // Platform information
  getPlatform,
  getPlatformName,

  // Constants
  DIST_DIR,
  BIN_DIR,
};
