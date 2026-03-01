# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Code search module for the SYMFLUENCE AI agent.

Provides ripgrep-based code search capabilities for finding patterns,
definitions, and usages across the codebase.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


class CodeSearch:
    """
    Code search using ripgrep for fast, regex-capable searching.

    Falls back to grep if ripgrep is not available.
    """

    def __init__(self, repo_root: Optional[str] = None):
        """
        Initialize code search.

        Args:
            repo_root: Repository root directory. If None, uses current directory.
        """
        import os
        if repo_root is None:
            repo_root = os.getcwd()
        self.repo_root = Path(repo_root).resolve()
        self._rg_available = shutil.which('rg') is not None
        self._grep_available = shutil.which('grep') is not None

    def search(
        self,
        pattern: str,
        file_glob: str = "*.py",
        context_lines: int = 2,
        max_results: int = 50,
        case_sensitive: bool = True,
        whole_word: bool = False,
        include_hidden: bool = False
    ) -> Tuple[bool, str]:
        """
        Search for a pattern in the codebase.

        Args:
            pattern: Regex pattern to search for
            file_glob: File glob pattern (e.g., "*.py", "*.yaml")
            context_lines: Number of context lines before/after match
            max_results: Maximum number of results to return
            case_sensitive: Whether search is case sensitive
            whole_word: Match whole words only
            include_hidden: Include hidden files/directories

        Returns:
            Tuple of (success, results_or_error)
        """
        try:
            if self._rg_available:
                return self._search_with_ripgrep(
                    pattern, file_glob, context_lines, max_results,
                    case_sensitive, whole_word, include_hidden
                )
            elif self._grep_available:
                return self._search_with_grep(
                    pattern, file_glob, context_lines, max_results,
                    case_sensitive, whole_word
                )
            else:
                return self._search_with_python(
                    pattern, file_glob, context_lines, max_results,
                    case_sensitive
                )
        except subprocess.TimeoutExpired:
            return False, "Search timed out"
        except subprocess.CalledProcessError as e:
            return False, f"Search command failed: {str(e)}"
        except (FileNotFoundError, PermissionError, OSError) as e:
            return False, f"Search failed: {str(e)}"

    def find_definition(
        self,
        name: str,
        definition_type: str = "any"
    ) -> Tuple[bool, str]:
        """
        Find the definition of a function, class, or variable.

        Args:
            name: Name of the symbol to find
            definition_type: "function", "class", "variable", or "any"

        Returns:
            Tuple of (success, results_or_error)
        """
        patterns = {
            "function": rf"^\s*def\s+{name}\s*\(",
            "class": rf"^\s*class\s+{name}\s*[\(:]",
            "variable": rf"^\s*{name}\s*=",
            "any": rf"^\s*(def|class)\s+{name}\s*[\(:]|^\s*{name}\s*="
        }

        pattern = patterns.get(definition_type, patterns["any"])
        return self.search(pattern, file_glob="*.py", context_lines=5)

    def find_usages(
        self,
        name: str,
        file_glob: str = "*.py"
    ) -> Tuple[bool, str]:
        """
        Find all usages of a symbol.

        Args:
            name: Name of the symbol to find usages of
            file_glob: File glob pattern

        Returns:
            Tuple of (success, results_or_error)
        """
        # Use word boundary matching to avoid partial matches
        pattern = rf"\b{name}\b"
        return self.search(pattern, file_glob=file_glob, context_lines=1, whole_word=True)

    def find_imports(
        self,
        module_name: str
    ) -> Tuple[bool, str]:
        """
        Find all imports of a module.

        Args:
            module_name: Name of the module to find imports of

        Returns:
            Tuple of (success, results_or_error)
        """
        # Match both "import module" and "from module import ..."
        pattern = rf"^\s*(from\s+{module_name}|import\s+{module_name})"
        return self.search(pattern, file_glob="*.py", context_lines=0)

    def find_files(
        self,
        pattern: str,
        file_type: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Find files matching a pattern.

        Args:
            pattern: Glob or regex pattern for file names
            file_type: Optional file extension filter (e.g., "py", "yaml")

        Returns:
            Tuple of (success, results_or_error)
        """
        try:
            if self._rg_available:
                cmd = ["rg", "--files", "-g", f"*{pattern}*"]
                if file_type:
                    cmd.extend(["-g", f"*.{file_type}"])
            else:
                # Fall back to find command
                cmd = ["find", ".", "-name", f"*{pattern}*", "-type", "f"]
                if file_type:
                    cmd = ["find", ".", "-name", f"*{pattern}*.{file_type}", "-type", "f"]

            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode not in (0, 1):
                return False, f"Find failed: {result.stderr}"

            output = result.stdout.strip()
            if not output:
                return True, "No files found matching pattern"

            # Format output
            files = output.split('\n')
            formatted = f"Found {len(files)} file(s):\n\n"
            for f in files[:100]:  # Limit to 100 files
                formatted += f"  {f}\n"
            if len(files) > 100:
                formatted += f"\n  ... and {len(files) - 100} more files"

            return True, formatted

        except subprocess.TimeoutExpired:
            return False, "Search timed out"
        except subprocess.CalledProcessError as e:
            return False, f"Find command failed: {str(e)}"
        except (FileNotFoundError, PermissionError, OSError) as e:
            return False, f"Find files failed: {str(e)}"

    def _search_with_ripgrep(
        self,
        pattern: str,
        file_glob: str,
        context_lines: int,
        max_results: int,
        case_sensitive: bool,
        whole_word: bool,
        include_hidden: bool
    ) -> Tuple[bool, str]:
        """Search using ripgrep."""
        cmd = [
            "rg",
            "--line-number",
            "--with-filename",
            f"--context={context_lines}",
            f"--max-count={max_results}",
            "--color=never"
        ]

        if not case_sensitive:
            cmd.append("--ignore-case")
        if whole_word:
            cmd.append("--word-regexp")
        if include_hidden:
            cmd.append("--hidden")
        if file_glob:
            cmd.extend(["--glob", file_glob])

        cmd.append(pattern)

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=30
        )

        # ripgrep returns 1 if no matches found (not an error)
        if result.returncode not in (0, 1):
            return False, f"Search failed: {result.stderr}"

        output = result.stdout.strip()
        if not output:
            return True, "No matches found"

        # Format output with summary
        lines = output.split('\n')
        match_count = len([l for l in lines if l and not l.startswith('--')])
        formatted = f"Found {match_count} match(es):\n\n{output}"

        return True, formatted

    def _search_with_grep(
        self,
        pattern: str,
        file_glob: str,
        context_lines: int,
        max_results: int,
        case_sensitive: bool,
        whole_word: bool
    ) -> Tuple[bool, str]:
        """Search using grep (fallback)."""
        cmd = [
            "grep",
            "-r",
            "-n",
            f"-C{context_lines}",
            f"-m{max_results}"
        ]

        if not case_sensitive:
            cmd.append("-i")
        if whole_word:
            cmd.append("-w")
        if file_glob:
            cmd.extend(["--include", file_glob])

        cmd.extend([pattern, "."])

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode not in (0, 1):
            return False, f"Search failed: {result.stderr}"

        output = result.stdout.strip()
        if not output:
            return True, "No matches found"

        return True, output

    def _search_with_python(
        self,
        pattern: str,
        file_glob: str,
        context_lines: int,
        max_results: int,
        case_sensitive: bool
    ) -> Tuple[bool, str]:
        """Pure Python fallback search."""
        import fnmatch
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return False, f"Invalid regex pattern: {e}"

        results = []
        match_count = 0

        for py_file in self.repo_root.rglob("*"):
            if not py_file.is_file():
                continue
            if file_glob and not fnmatch.fnmatch(py_file.name, file_glob):
                continue
            if any(part.startswith('.') for part in py_file.parts):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    if regex.search(line):
                        match_count += 1
                        if match_count > max_results:
                            break

                        # Get context
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)

                        rel_path = py_file.relative_to(self.repo_root)
                        results.append(f"\n{rel_path}:{i+1}:")
                        for j in range(start, end):
                            prefix = ">" if j == i else " "
                            results.append(f"{prefix} {j+1}: {lines[j].rstrip()}")

                if match_count > max_results:
                    break

            except (OSError, UnicodeDecodeError, SyntaxError):
                continue  # Skip unreadable or non-Python files

        if not results:
            return True, "No matches found"

        output = f"Found {match_count} match(es):\n" + "\n".join(results)
        return True, output


class FuzzyMatcher:
    """
    Fuzzy code matching for edit operations.

    Helps find approximate matches when exact string matching fails.
    """

    @staticmethod
    def find_best_match(
        content: str,
        target: str,
        threshold: float = 0.8
    ) -> Tuple[Optional[int], Optional[str], float]:
        """
        Find the best matching location for target text in content.

        Args:
            content: Full file content to search in
            target: Code snippet to find
            threshold: Minimum similarity ratio (0.0 to 1.0)

        Returns:
            Tuple of (start_line, matched_text, similarity_ratio)
            Returns (None, None, 0.0) if no match above threshold
        """
        from difflib import SequenceMatcher

        content_lines = content.split('\n')
        target_lines = target.split('\n')
        target_len = len(target_lines)

        best_match = (None, None, 0.0)

        # Sliding window search
        for i in range(len(content_lines) - target_len + 1):
            candidate = '\n'.join(content_lines[i:i + target_len])

            # Calculate similarity
            ratio = SequenceMatcher(None, candidate, target).ratio()

            if ratio > best_match[2]:
                best_match = (i + 1, candidate, ratio)  # 1-indexed line

        if best_match[2] >= threshold:
            return best_match

        return None, None, 0.0

    @staticmethod
    def normalize_whitespace(code: str) -> str:
        """
        Normalize whitespace for comparison.

        Args:
            code: Code string to normalize

        Returns:
            Normalized code string
        """
        lines = code.split('\n')
        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in lines]
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines)

    @staticmethod
    def get_indentation(code: str) -> str:
        """
        Get the common indentation of a code block.

        Args:
            code: Code string

        Returns:
            Common indentation string
        """
        lines = [l for l in code.split('\n') if l.strip()]
        if not lines:
            return ""

        # Find minimum indentation
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        if min_indent == float('inf'):
            return ""

        return lines[0][:int(min_indent)]

    @staticmethod
    def apply_indentation(code: str, indent: str) -> str:
        """
        Apply indentation to a code block.

        Args:
            code: Code string
            indent: Indentation to apply

        Returns:
            Indented code string
        """
        lines = code.split('\n')
        result = []
        for line in lines:
            if line.strip():
                result.append(indent + line.lstrip())
            else:
                result.append(line)
        return '\n'.join(result)
