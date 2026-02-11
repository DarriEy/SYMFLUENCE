/**
 * Platform detection and mapping for SYMFLUENCE binaries
 */

// Mapping from Node.js platform/arch to SYMFLUENCE release naming.
// Only platforms with pre-built binaries in CI (linux-x86_64, macos-arm64).
const PLATFORM_MAP = {
  'darwin-arm64': 'macos-arm64',
  'linux-x64': 'linux-x86_64',
};

/**
 * Get the current platform identifier for SYMFLUENCE releases
 * @returns {string} Platform identifier (e.g., 'macos-arm64', 'linux-x86_64')
 * @throws {Error} If platform is not supported
 */
function getPlatform() {
  const platform = process.platform;
  const arch = process.arch;
  const key = `${platform}-${arch}`;

  if (!PLATFORM_MAP[key]) {
    const name = getPlatformName();
    throw new Error(
      `Unsupported platform: ${name} (${platform} ${arch})\n` +
      `Pre-built binaries are available for: Linux x86_64, macOS Apple Silicon\n` +
      `You can build from source instead: symfluence binary install\n` +
      `See: https://github.com/DarriEy/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md`
    );
  }

  return PLATFORM_MAP[key];
}

/**
 * Check if current platform is supported
 * @returns {boolean} True if platform is supported
 */
function isPlatformSupported() {
  const platform = process.platform;
  const arch = process.arch;
  const key = `${platform}-${arch}`;
  return key in PLATFORM_MAP;
}

/**
 * Get user-friendly platform name
 * @returns {string} Platform name for display
 */
function getPlatformName() {
  const platform = process.platform;
  const arch = process.arch;

  const names = {
    'darwin-arm64': 'macOS (Apple Silicon)',
    'darwin-x64': 'macOS (Intel)',
    'linux-x64': 'Linux (x86_64)',
    'linux-arm64': 'Linux (ARM64)',
  };

  return names[`${platform}-${arch}`] || `${platform} ${arch}`;
}

/**
 * Get all supported platforms
 * @returns {Array<string>} List of supported platform identifiers
 */
function getSupportedPlatforms() {
  return Object.values(PLATFORM_MAP);
}

module.exports = {
  getPlatform,
  isPlatformSupported,
  getPlatformName,
  getSupportedPlatforms,
  PLATFORM_MAP,
};
