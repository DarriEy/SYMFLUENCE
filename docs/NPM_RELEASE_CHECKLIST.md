# SYMFLUENCE npm Release Checklist

## Version Management

### Version Synchronization
SYMFLUENCE has **two version sources** that must be kept in sync:

1. **`pyproject.toml`** (line 10): Python package version
2. **`npm/package.json`** (line 3): npm CLI package version

**CRITICAL**: Both must match the release tag exactly.

### Pre-Release Version Update
```bash
# 1. Determine new version (e.g., 0.5.11)
NEW_VERSION="0.5.11"

# 2. Update pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml

# 3. Update npm/package.json
cd npm && npm version $NEW_VERSION --no-git-tag-version && cd ..

# 4. Verify both are updated
grep "^version = " pyproject.toml
grep "\"version\":" npm/package.json

# 5. Commit the version bump
git add pyproject.toml npm/package.json
git commit -m "chore: bump version to $NEW_VERSION"
git push origin develop
```

### Version Validation Script
Create `scripts/check_version_sync.sh` to validate:
```bash
#!/bin/bash
PYPROJECT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
NPM_VERSION=$(grep '"version":' npm/package.json | sed 's/.*"\([0-9.]*\)".*/\1/')

if [ "$PYPROJECT_VERSION" != "$NPM_VERSION" ]; then
    echo "❌ Version mismatch!"
    echo "   pyproject.toml: $PYPROJECT_VERSION"
    echo "   npm/package.json: $NPM_VERSION"
    exit 1
else
    echo "✓ Versions in sync: $PYPROJECT_VERSION"
fi
```

## Pre-Release Checklist

### 1. Version Preparation
- [ ] **Run version synchronization script** (see above)
- [ ] Update `pyproject.toml` version to match release tag
- [ ] Update `npm/package.json` version to match release tag
- [ ] **Verify both versions are identical**
- [ ] Ensure version follows semver (MAJOR.MINOR.PATCH)
- [ ] Update CHANGELOG.md with release notes

### 2. Code Quality
- [ ] All tests passing (`pytest tests/`)
- [ ] No linting errors
- [ ] Documentation updated

### 3. npm Package Testing
```bash
cd npm/
npm pack
npm install -g ./symfluence-*.tgz --force
symfluence help
```

### 4. GitHub Secrets
- [ ] `NPM_TOKEN` secret configured in repository
- [ ] Token is granular access token with:
  - Read and write permissions
  - Automation enabled (bypasses 2FA)
  - 90-day expiration (set calendar reminder)

## Release Process

### Creating a Release

```bash
# 1. Ensure you're on develop branch with latest changes
git checkout develop
git pull origin develop

# 2. Create and push the release
gh release create vX.Y.Z \
  --target develop \
  --title "SYMFLUENCE vX.Y.Z" \
  --notes "Release notes here"

# 3. Monitor the workflow
gh run list --workflow=release-binaries.yml --limit 1
gh run watch  # Watch the latest run
```

### What Happens Automatically

The `release-binaries.yml` workflow will:

1. **Build Phase** (15-20 minutes)
   - Build binaries for Linux x86_64
   - Build binaries for macOS ARM64
   - Generate toolchain.json with build metadata
   - Create tarballs with SHA256 checksums

2. **Upload Phase**
   - Upload binaries to GitHub release
   - Store artifacts (90-day retention)

3. **npm Publish Phase**
   - Verify package version matches release tag
   - Publish to npm registry (if not a prerelease)
   - Add release summary

## Post-Release Verification

### 1. Check GitHub Release
```bash
gh release view vX.Y.Z
```

Should show:
- `symfluence-tools-vX.Y.Z-linux-x86_64.tar.gz`
- `symfluence-tools-vX.Y.Z-linux-x86_64.tar.gz.sha256`
- `symfluence-tools-vX.Y.Z-macos-arm64.tar.gz`
- `symfluence-tools-vX.Y.Z-macos-arm64.tar.gz.sha256`

### 2. Verify npm Publication
```bash
npm view symfluence
npm view symfluence@X.Y.Z
```

### 3. Test End-to-End Installation

#### Linux (Ubuntu 22.04+)
```bash
# Fresh VM/container
npm install -g symfluence
symfluence info
symfluence --doctor

# Should show all tools installed
```

#### macOS (Apple Silicon)
```bash
# Fresh system or new user
npm install -g symfluence
symfluence info
symfluence --doctor
```

### 4. Test Binary Functionality
```bash
# Get binary directory
BIN_DIR=$(npm root -g)/symfluence/dist/bin

# Test SUMMA
$BIN_DIR/summa --version

# Test mizuRoute
$BIN_DIR/mizuroute --help

# Test NGEN
$BIN_DIR/ngen --version
```

## Troubleshooting

### Build Fails

**Check workflow logs:**
```bash
gh run view --log-failed
```

**Common issues:**
- Disk space (Linux runner)
- Compilation errors (check compiler versions)
- Missing dependencies (NetCDF, HDF5)

**Solution:** Fix issue, delete release, recreate:
```bash
gh release delete vX.Y.Z --yes
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z
# Then recreate release
```

### npm Publish Fails

**Check for:**
- NPM_TOKEN expired (90-day limit)
- Version already published (can't republish same version)
- Token lacks permissions

**View npm publish logs:**
```bash
gh run view --log | grep -A 20 "Publish to npm"
```

**Solution:**
- Renew token if expired
- Bump version and create new release
- Verify token has "Automation" enabled

### Binary Download Fails After npm Install

**Verify binaries exist:**
```bash
gh release view vX.Y.Z --json assets
```

**Common issues:**
- Version mismatch (npm package vs release tag)
- Release is a prerelease (binaries may not be built)
- 404 errors (release doesn't exist)

**Solution:**
- Ensure npm/package.json version matches release tag
- Verify release is not marked as prerelease
- Check release artifacts exist

## Token Maintenance

### Renewing npm Token (Every 90 Days)

1. **Create new token:**
   - https://www.npmjs.com/settings/YOUR_USERNAME/tokens/granular-access-token/new
   - Name: `SYMFLUENCE_GITHUB_ACTIONS`
   - Expiration: 90 days
   - Permissions: Read and write
   - **Enable Automation**

2. **Update GitHub secret:**
   - https://github.com/DarriEy/SYMFLUENCE/settings/secrets/actions
   - Update `NPM_TOKEN` value

3. **Set reminder:**
   - Calendar reminder for 80 days from now
   - Renew before expiration to avoid disruption

### Token Expiration Monitoring

Check token status:
- npm dashboard: https://www.npmjs.com/settings/YOUR_USERNAME/tokens
- Test token: `NPM_TOKEN=xxx npm publish --dry-run`

## Release Cadence

### Version Numbering

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Pre-releases

For testing before official release:

```bash
# Version: 0.5.9-beta.1
gh release create v0.5.9-beta.1 --prerelease --notes "Beta release"

# npm will publish as beta tag
npm view symfluence@beta
```

## Rollback Procedure

If a release has critical issues:

### 1. Deprecate npm Version
```bash
npm deprecate symfluence@X.Y.Z "Critical bug, use X.Y.Z+1 instead"
```

### 2. Delete GitHub Release (if needed)
```bash
gh release delete vX.Y.Z --yes
```

### 3. Create Fixed Version
- Increment patch version
- Create new release with fixes

## Monitoring

### After Release

- Monitor npm downloads: https://www.npmjs.com/package/symfluence
- Watch for issues: https://github.com/DarriEy/SYMFLUENCE/issues
- Check discussions: https://github.com/DarriEy/SYMFLUENCE/discussions

### Success Metrics

- [ ] Binaries build successfully for both platforms
- [ ] npm package publishes without errors
- [ ] Installation works on both Linux and macOS
- [ ] All CLI commands functional
- [ ] No critical issues reported in first 24 hours

## Quick Reference

### URLs
- npm package: https://www.npmjs.com/package/symfluence
- Releases: https://github.com/DarriEy/SYMFLUENCE/releases
- Workflows: https://github.com/DarriEy/SYMFLUENCE/actions
- Secrets: https://github.com/DarriEy/SYMFLUENCE/settings/secrets/actions

### Commands
```bash
# Create release
gh release create vX.Y.Z --target develop

# Watch build
gh run watch

# Verify npm
npm view symfluence@X.Y.Z

# Test install
npm install -g symfluence@X.Y.Z

# Check binaries
symfluence info
```

---

**Last Updated**: 2025-12-31
**Maintained By**: Repository maintainers
