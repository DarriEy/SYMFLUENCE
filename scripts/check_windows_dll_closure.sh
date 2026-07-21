#!/usr/bin/env bash
#
# Verify that every Windows executable/DLL in a staged SYMFLUENCE tool tree can
# resolve all of the DLLs it imports.
#
# ---------------------------------------------------------------------------
# Why this exists
# ---------------------------------------------------------------------------
# A packaged tool that is missing one imported DLL does NOT fail loudly on
# Windows: the loader aborts the process with STATUS_DLL_NOT_FOUND (0xC0000135)
# before main() runs, so the program produces *no stdout and no stderr* and
# exits in ~0 seconds. SYMFLUENCE's calibration framework records that as a
# -9999 objective — indistinguishable from a model that ran and scored badly.
# Seven native HYPE calibrations were silently lost that way, because `hype`
# imports libnetcdff-7.dll and the package did not ship it.
#
# So packaging must prove, statically, that the import closure is satisfied.
# This walks each PE image's import table with objdump and checks every imported
# DLL against (a) the staged directories and (b) the Windows system DLLs. No
# hand-maintained per-tool DLL lists.
#
# Usage:
#   scripts/check_windows_dll_closure.sh <dir> [<dir> ...]
#
# All supplied directories form one search set (typically the staged bin/ and
# lib/). Exits non-zero if any import is unresolvable.
#
set -uo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dir> [<dir> ...]" >&2
    exit 2
fi

if ! command -v objdump >/dev/null 2>&1; then
    echo "::warning::objdump not available — skipping Windows DLL closure check"
    exit 0
fi

# Callers may hand us native Windows paths (e.g. `symfluence path` output).
# Normalise them so find/ls/objdump see a POSIX path.
DIRS=()
for _d in "$@"; do
    if command -v cygpath >/dev/null 2>&1; then
        _d="$(cygpath -u "$_d" 2>/dev/null || printf '%s' "$_d")"
    fi
    DIRS+=("$_d")
done

# DLLs that are always Windows-provided. The primary test is "does a file with
# this name exist in the system directory"; this pattern covers the cases that
# test cannot see: the virtual API-set DLLs (api-ms-*/ext-ms-*) have no file on
# disk at all, and MS-MPI is deliberately not bundled (users install the MS-MPI
# redistributable). It is also the fallback when the system directory cannot be
# enumerated.
SYSTEM_DLL_RE='^(api-ms-.*|ext-ms-.*|msmpi|kernel32|kernelbase|ntdll|msvcrt|user32|advapi32|ws2_32|shell32|shlwapi|ole32|oleaut32|gdi32|crypt32|secur32|bcrypt|rpcrt4|sechost|combase|ucrtbase|dbghelp|version|imm32|setupapi|userenv|iphlpapi|dnsapi|winmm|comdlg32|comctl32|powrprof|psapi|wsock32|mpr|wldap32|normaliz|netapi32|pdh|wtsapi32|dwmapi|uxtheme)\.dll$'

# Build a lowercase name index ONCE, in memory. Shelling out per lookup (a
# `find` over System32's thousands of entries, or a grep over an index file)
# turns this check into minutes; associative-array lookups keep it seconds.
declare -A AVAILABLE=()
SYSTEM_SEEN=0

for d in "${DIRS[@]}"; do
    [ -d "$d" ] || continue
    while IFS= read -r n; do
        [ -n "$n" ] && AVAILABLE["${n,,}"]=staged
    done < <(ls -1 "$d" 2>/dev/null)
done

_sysroot="${SYSTEMROOT:-${SystemRoot:-C:\\Windows}}"
if command -v cygpath >/dev/null 2>&1; then
    _sysroot="$(cygpath -u "$_sysroot" 2>/dev/null || echo "/c/Windows")"
fi
for d in "$_sysroot/System32" "$_sysroot/SysWOW64"; do
    [ -d "$d" ] || continue
    while IFS= read -r n; do
        [ -n "$n" ] || continue
        AVAILABLE["${n,,}"]=system
        SYSTEM_SEEN=1
    done < <(ls -1 "$d" 2>/dev/null)
done

if [ "$SYSTEM_SEEN" -eq 0 ]; then
    echo "::warning::Could not enumerate $_sysroot — system DLLs matched by name pattern only"
fi

# Collect the PE images to inspect. Detect by CONTENT, not by extension:
# SYMFLUENCE stages Windows executables under their unsuffixed tool names
# (bin/hype, bin/summa, ...), so an "*.exe" glob would miss almost every model —
# which is exactly how libnetcdff-7.dll went unbundled.
PE_FILES=()
for d in "${DIRS[@]}"; do
    [ -d "$d" ] || continue
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        [ -L "$f" ] && continue
        [ "$(head -c 2 "$f" 2>/dev/null)" = "MZ" ] || continue
        PE_FILES+=("$f")
    done < <(find "$d" -maxdepth 1 -type f 2>/dev/null | sort)
done

if [ "${#PE_FILES[@]}" -eq 0 ]; then
    echo "::warning::No PE images found in: ${DIRS[*]}"
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Windows DLL import-closure check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Search dirs: ${DIRS[*]}"
echo "PE images:   ${#PE_FILES[@]}"
echo ""

BROKEN=0
for pe in "${PE_FILES[@]}"; do
    pe_name="$(basename "$pe")"
    missing=""
    # `objdump -p` prints the import table before the export table; quitting at
    # the export section avoids dumping thousands of export names per DLL, which
    # otherwise dominates the runtime on libnetcdf/libgdal.
    while IFS= read -r dll; do
        [ -n "$dll" ] || continue
        [ -n "${AVAILABLE[$dll]:-}" ] && continue
        [[ "$dll" =~ $SYSTEM_DLL_RE ]] && continue
        missing="$missing $dll"
    done < <(objdump -p "$pe" 2>/dev/null \
        | awk '/Export Address Table/ {exit} /DLL Name:/ {print tolower($3)}' | sort -u)

    if [ -n "$missing" ]; then
        echo "::error::$pe_name imports DLLs that are neither bundled nor system-provided:$missing"
        BROKEN=$((BROKEN + 1))
    else
        echo "  ok  $pe_name"
    fi
done

echo ""
if [ "$BROKEN" -gt 0 ]; then
    echo "::error::$BROKEN packaged binaries have an unsatisfied DLL import closure."
    echo "         On Windows these abort with 0xC0000135 before main(), emitting"
    echo "         no output at all — which calibration records as a -9999 score."
    exit 1
fi

echo "All ${#PE_FILES[@]} PE images have a fully satisfied import closure."
