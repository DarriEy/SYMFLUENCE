"""
PR management module for staging and proposing code changes.

Generates PR titles, descriptions, and manages git staging.
Supports fuzzy matching for code edits and automated PR creation via gh CLI.
"""

import re
import shutil
import subprocess
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple


class PRManager:
    """Manage PR proposals and change staging with SOTA capabilities."""

    # Attribution for agent-generated commits
    AGENT_CO_AUTHOR = "Co-Authored-By: SYMFLUENCE Agent <agent@symfluence.dev>"

    def __init__(self, repo_root: Optional[str] = None):
        """
        Initialize PR manager.

        Args:
            repo_root: Repository root directory. If None, uses current directory.
        """
        if repo_root is None:
            import os
            repo_root = os.getcwd()
        self.repo_root = Path(repo_root).resolve()
        self._gh_available = shutil.which('gh') is not None

    def propose_code_change(
        self,
        file_path: str,
        old_code: str,
        new_code: str,
        description: str,
        reason: str = "improvement",
        fuzzy_threshold: float = 0.85
    ) -> Tuple[bool, str]:
        """
        Propose a code modification with validation, fuzzy matching, and staging.

        Uses fuzzy matching to find the target code even if whitespace or minor
        differences exist. Falls back to exact matching if fuzzy match is ambiguous.

        Args:
            file_path: Path to file relative to repo root
            old_code: Code to replace (supports fuzzy matching)
            new_code: Replacement code
            description: Description of why this change is needed
            reason: Type of change (bugfix, improvement, feature)
            fuzzy_threshold: Minimum similarity for fuzzy match (0.0-1.0).
                Default 0.85 (85%) allows minor whitespace/formatting differences
                while preventing incorrect matches. Lower values (0.7-0.8) for
                more flexible matching, higher values (0.9+) for stricter matching.

        Returns:
            Tuple of (success, message_or_error)
        """
        try:
            from symfluence.agent.file_operations import FileOperations

            file_ops = FileOperations(str(self.repo_root))

            # Read the current file (without line numbers for processing)
            full_path = self.repo_root / file_path
            if not full_path.exists():
                return False, f"File not found: {file_path}"

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Normalize whitespace for comparison
            old_code_normalized = self._normalize_code(old_code)
            content_normalized = self._normalize_code(content)

            # Try exact match first
            if old_code in content:
                modified_content = content.replace(old_code, new_code, 1)
                match_info = "exact match"
            elif old_code_normalized in content_normalized:
                # Exact match after normalization - find and replace carefully
                match_start, match_end = self._find_normalized_match(content, old_code)
                if match_start is not None:
                    modified_content = content[:match_start] + new_code + content[match_end:]
                    match_info = "normalized match"
                else:
                    return False, "Match found but replacement failed"
            else:
                # Try fuzzy matching
                match_result = self._fuzzy_find_and_replace(
                    content, old_code, new_code, fuzzy_threshold
                )
                if match_result[0]:
                    modified_content = match_result[1]
                    match_info = f"fuzzy match (similarity: {match_result[2]:.1%})"
                else:
                    # No match found - provide helpful error
                    return False, self._generate_match_error(content, old_code, file_path)

            if modified_content == content:
                return False, "Replacement resulted in no changes"

            # Validate Python syntax if applicable
            if file_path.endswith('.py'):
                syntax_ok, syntax_err = self._validate_python_syntax(modified_content)
                if not syntax_ok:
                    return False, f"Syntax error in modified code: {syntax_err}"

            # Write modified content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)

            # Show the diff
            success, diff = file_ops.show_diff(file_path)

            output = f"✓ Change proposed to {file_path} ({match_info})\n\n"
            output += f"Reason: {reason}\n"
            output += f"Description: {description}\n\n"
            output += "Diff:\n"
            output += "=" * 60 + "\n"
            output += diff if success else "(no diff available)"

            # Stage the changes
            file_ops.stage_changes([file_path])

            output += "\n" + "=" * 60 + "\n"
            output += "✓ Changes staged to git\n\n"
            output += "Next steps:\n"
            output += "1. Review the diff above\n"
            output += "2. Run tests to validate\n"
            output += "3. Use create_pr or create_pr_proposal to complete\n"

            return True, output

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error proposing code change: {str(e)}"

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison (strip trailing whitespace, normalize line endings)."""
        lines = code.split('\n')
        normalized = [line.rstrip() for line in lines]
        # Remove leading/trailing empty lines
        while normalized and not normalized[0].strip():
            normalized.pop(0)
        while normalized and not normalized[-1].strip():
            normalized.pop()
        return '\n'.join(normalized)

    def _find_normalized_match(
        self,
        content: str,
        target: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find the position of target in content after normalization."""
        content_lines = content.split('\n')
        target_lines = self._normalize_code(target).split('\n')

        for i in range(len(content_lines) - len(target_lines) + 1):
            candidate_lines = content_lines[i:i + len(target_lines)]
            candidate_normalized = [l.rstrip() for l in candidate_lines]
            if candidate_normalized == target_lines:
                # Found match - calculate byte positions
                start = sum(len(l) + 1 for l in content_lines[:i])
                end = start + sum(len(l) + 1 for l in candidate_lines) - 1
                return start, end

        return None, None

    def _fuzzy_find_and_replace(
        self,
        content: str,
        old_code: str,
        new_code: str,
        threshold: float
    ) -> Tuple[bool, str, float]:
        """
        Find and replace using fuzzy matching.

        Returns:
            Tuple of (success, modified_content, similarity_ratio)
        """
        content_lines = content.split('\n')
        old_lines = old_code.split('\n')
        old_len = len(old_lines)

        best_match: tuple[int, int, float] = (-1, -1, 0.0)  # (start_idx, end_idx, ratio)

        # Sliding window search
        for i in range(len(content_lines) - old_len + 1):
            candidate = '\n'.join(content_lines[i:i + old_len])
            ratio = SequenceMatcher(None, candidate, old_code).ratio()

            if ratio > best_match[2]:
                best_match = (i, i + old_len, ratio)

        if best_match[2] >= threshold:
            start_idx, end_idx, ratio = best_match

            # Preserve indentation from original
            original_indent = self._get_indentation(content_lines[start_idx])
            new_code_adjusted = self._apply_indentation(new_code, original_indent)

            # Perform replacement
            new_lines = new_code_adjusted.split('\n')
            result_lines = content_lines[:start_idx] + new_lines + content_lines[end_idx:]

            return True, '\n'.join(result_lines), ratio

        return False, content, 0.0

    def _get_indentation(self, line: str) -> str:
        """Get the leading whitespace of a line."""
        return line[:len(line) - len(line.lstrip())]

    def _apply_indentation(self, code: str, base_indent: str) -> str:
        """Apply base indentation to a code block while preserving relative indentation."""
        lines = code.split('\n')
        if not lines:
            return code

        # Find the minimum indentation in the new code
        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            return code

        min_indent = min(len(l) - len(l.lstrip()) for l in non_empty_lines)

        # Apply new indentation
        result = []
        for line in lines:
            if line.strip():
                relative_indent = len(line) - len(line.lstrip()) - min_indent
                result.append(base_indent + ' ' * relative_indent + line.lstrip())
            else:
                result.append(line)

        return '\n'.join(result)

    def _generate_match_error(self, content: str, old_code: str, file_path: str) -> str:
        """Generate a helpful error message when code isn't found."""
        # Try to find similar code
        content_lines = content.split('\n')
        old_lines = old_code.split('\n')

        # Look for partial matches (first line of old_code)
        first_line = old_lines[0].strip() if old_lines else ""
        similar_locations = []

        for i, line in enumerate(content_lines):
            if first_line and first_line in line:
                similar_locations.append((i + 1, line.strip()[:60]))

        error_msg = f"Code not found in {file_path}.\n\n"
        error_msg += f"Looking for ({len(old_lines)} lines starting with):\n"
        error_msg += f"  {old_lines[0][:80] if old_lines else '(empty)'}...\n\n"

        if similar_locations:
            error_msg += "Similar code found at:\n"
            for line_num, preview in similar_locations[:5]:
                error_msg += f"  Line {line_num}: {preview}...\n"
            error_msg += "\nTry reading the file first to get exact code."
        else:
            error_msg += "No similar code found. Please read the file first to verify the code exists."

        return error_msg

    def _validate_python_syntax(self, content: str) -> Tuple[bool, str]:
        """Validate Python code syntax."""
        try:
            compile(content, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def show_staged_changes(self) -> Tuple[bool, str]:
        """
        Display all staged changes with statistics.

        Returns:
            Tuple of (success, diff_or_error)
        """
        try:
            from symfluence.agent.file_operations import FileOperations

            file_ops = FileOperations(str(self.repo_root))
            success, diff = file_ops.get_staged_changes()

            if not success:
                return False, diff

            if not diff or diff == "(no changes)":
                return True, "No staged changes\n\nUse propose_code_change to stage modifications."

            # Get diff statistics
            stat_result = self._run_git(["diff", "--cached", "--stat"])
            stats = stat_result.stdout.strip() if stat_result.returncode == 0 else ""

            # Get list of modified files
            success_files, files = self.get_modified_files()
            file_count = len(files) if success_files else 0

            output = "Staged Changes\n" + "=" * 60 + "\n\n"

            # Show statistics summary
            if stats:
                output += "Summary:\n"
                output += stats + "\n\n"

            output += f"Files changed: {file_count}\n"
            if success_files and files:
                for f in files:
                    output += f"  - {f}\n"
                output += "\n"

            output += "Diff:\n"
            output += "-" * 60 + "\n"
            output += diff + "\n"
            output += "-" * 60 + "\n\n"

            output += "Next steps:\n"
            output += "  - Review the changes above\n"
            output += "  - Run tests: pytest -v -m unit\n"
            output += "  - Commit: git commit -m 'message'\n"
            output += "  - Or unstage: git reset\n"

            return True, output

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error showing staged changes: {str(e)}"

    def generate_pr_title(
        self,
        change_description: str,
        reason: str = "improvement"
    ) -> str:
        """
        Generate a good PR title from change description.

        Args:
            change_description: Description of the change
            reason: Type of change (bugfix, improvement, feature)

        Returns:
            Generated PR title
        """
        # Capitalize first letter
        title = change_description[0].upper() + change_description[1:] if change_description else ""

        # Remove trailing period if present
        if title.endswith('.'):
            title = title[:-1]

        # Add prefix based on reason
        if reason == "bugfix":
            title = f"Fix: {title}"
        elif reason == "feature":
            title = f"Add: {title}"
        else:
            title = f"Improve: {title}" if not title.startswith(("Add", "Fix", "Improve")) else title

        return title[:72]  # GitHub limit

    def generate_pr_description(
        self,
        title: str,
        summary: str,
        reason: str = "improvement",
        files_modified: Optional[List[str]] = None,
        testing_notes: Optional[str] = None,
        auto_detect_files: bool = True
    ) -> str:
        """
        Generate a comprehensive PR description.

        Args:
            title: PR title
            summary: Summary of changes
            reason: Type of change
            files_modified: List of files modified (auto-detected if not provided)
            testing_notes: Notes on testing performed
            auto_detect_files: Auto-detect modified files from git staging

        Returns:
            Formatted PR description
        """
        description = f"## Summary\n{summary}\n\n"

        # Add context-specific reason section
        if reason == "bugfix":
            description += "## Problem & Solution\n"
            description += "This PR addresses an issue identified in the codebase. "
            description += "The fix involves changes described in the summary above.\n\n"
        elif reason == "feature":
            description += "## New Feature\n"
            description += "This PR introduces new functionality to SYMFLUENCE. "
            description += "See the summary above for details on what was added.\n\n"
        else:
            description += "## Changes\n"
            description += "This PR improves existing functionality in SYMFLUENCE. "
            description += "The changes enhance code quality, performance, or maintainability.\n\n"

        # Auto-detect files if not provided
        if files_modified is None and auto_detect_files:
            success, detected_files = self.get_modified_files()
            if success and detected_files:
                files_modified = detected_files

        # Files modified
        if files_modified:
            description += "## Files Modified\n"
            for file_path in files_modified:
                description += f"- `{file_path}`\n"
            description += "\n"

        # Testing section with guidance
        description += "## Testing\n"
        if testing_notes:
            description += f"{testing_notes}\n"
        else:
            description += "- [ ] All existing tests pass (`pytest -v -m unit`)\n"
            description += "- [ ] Changes reviewed for correctness\n"
            description += "- [ ] No breaking changes introduced\n"

        # Footer with attribution
        description += "\n---\n"
        description += "*Generated by SYMFLUENCE Agent*\n"
        description += f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*"

        return description

    def create_pr_proposal(
        self,
        title: str,
        description: str,
        branch_name: Optional[str] = None,
        reason: str = "improvement"
    ) -> Tuple[bool, str]:
        """
        Create a PR proposal by staging changes and preparing commit message.

        Args:
            title: PR title
            description: PR body
            branch_name: Optional branch name (default: auto-generated)
            reason: Type of change

        Returns:
            Tuple of (success, message_or_error)
        """
        try:
            # Get staged changes
            success, staged_output = self.show_staged_changes()
            if not success:
                return False, staged_output

            if "No staged changes" in staged_output:
                return False, "No staged changes to create PR. Use propose_code_change first."

            # Generate branch name if not provided
            if not branch_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                branch_name = f"agent-{reason}-{timestamp}"

            output = "PR Proposal Ready\n" + "=" * 60 + "\n\n"
            output += f"Title: {title}\n\n"
            output += "Staged changes are ready for commit.\n\n"

            output += "To complete this PR:\n"
            output += f"1. Create branch: git checkout -b {branch_name}\n"
            output += f"2. Commit changes: git commit -m '{self._escape_commit_message(title)}'\n"
            output += "3. Push branch: git push -u origin [branch-name]\n"
            output += "4. Create PR on GitHub: gh pr create\n\n"

            output += "Commit message:\n"
            output += "-" * 60 + "\n"
            output += f"{self._escape_commit_message(title)}\n\n{description}\n"
            output += "-" * 60 + "\n\n"

            output += "Staged changes:\n"
            output += staged_output

            return True, output

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error creating PR proposal: {str(e)}"

    def get_commit_log(self, max_commits: int = 10) -> Tuple[bool, str]:
        """
        Get recent commit log.

        Args:
            max_commits: Maximum number of commits to show

        Returns:
            Tuple of (success, log_or_error)
        """
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", f"-{max_commits}"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return False, f"Git error: {result.stderr}"

            return True, result.stdout or "No commits yet"

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error getting commit log: {str(e)}"

    def get_current_branch(self) -> Tuple[bool, str]:
        """
        Get current git branch.

        Returns:
            Tuple of (success, branch_name_or_error)
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return False, f"Git error: {result.stderr}"

            return True, result.stdout.strip()

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error getting branch: {str(e)}"

    def create_pr(
        self,
        title: str,
        description: str,
        branch_name: Optional[str] = None,
        base_branch: str = "main",
        reason: str = "improvement",
        draft: bool = False,
        auto_push: bool = True
    ) -> Tuple[bool, str]:
        """
        Create a complete PR automatically using gh CLI.

        This is the SOTA workflow that:
        1. Creates a new branch
        2. Commits staged changes with attribution
        3. Pushes to remote
        4. Creates PR via GitHub CLI

        Args:
            title: PR title
            description: PR body/description
            branch_name: Branch name (auto-generated if not provided)
            base_branch: Base branch to merge into (default: main)
            reason: Type of change (bugfix, improvement, feature)
            draft: Create as draft PR
            auto_push: Automatically push and create PR

        Returns:
            Tuple of (success, result_message_or_error)
        """
        try:
            # Verify gh CLI is available
            if not self._gh_available:
                return False, (
                    "GitHub CLI (gh) not found. Install it from https://cli.github.com/\n\n"
                    "Alternatively, use create_pr_proposal for manual PR creation."
                )

            # Check for staged changes
            success, staged = self.show_staged_changes()
            if not success or "No staged changes" in staged:
                return False, "No staged changes. Use propose_code_change first."

            # Generate branch name
            if not branch_name:
                slug = self._slugify(title)[:30]
                timestamp = datetime.now().strftime("%Y%m%d%H%M")
                branch_name = f"agent/{slug}-{timestamp}"

            output_parts = []

            # Step 1: Create and switch to new branch
            success, msg = self._create_branch(branch_name)
            if not success:
                return False, f"Failed to create branch: {msg}"
            output_parts.append(f"✓ Created branch: {branch_name}")

            # Step 2: Commit changes with attribution
            commit_msg = f"{title}\n\n{description}\n\n{self.AGENT_CO_AUTHOR}"
            success, msg = self._commit_changes(commit_msg)
            if not success:
                # Rollback branch creation
                self._run_git(["checkout", "-"])
                self._run_git(["branch", "-D", branch_name])
                return False, f"Failed to commit: {msg}"
            output_parts.append("✓ Committed changes")

            if auto_push:
                # Step 3: Push to remote
                success, msg = self._push_branch(branch_name)
                if not success:
                    output_parts.append(f"⚠ Push failed: {msg}")
                    output_parts.append("You can push manually with: git push -u origin " + branch_name)
                else:
                    output_parts.append(f"✓ Pushed to origin/{branch_name}")

                    # Step 4: Create PR via gh CLI
                    success, pr_result = self._create_github_pr(
                        title=title,
                        body=description,
                        base=base_branch,
                        draft=draft
                    )
                    if success:
                        output_parts.append(f"✓ PR created: {pr_result}")
                    else:
                        output_parts.append(f"⚠ PR creation failed: {pr_result}")
                        output_parts.append(f"Create manually with: gh pr create --title \"{title}\"")

            else:
                output_parts.append("\nTo complete PR creation:")
                output_parts.append(f"  git push -u origin {branch_name}")
                output_parts.append(f"  gh pr create --title \"{title}\" --base {base_branch}")

            return True, "\n".join(output_parts)

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error creating PR: {str(e)}"

    def _create_branch(self, branch_name: str) -> Tuple[bool, str]:
        """Create and switch to a new branch."""
        # Check if branch already exists
        result = self._run_git(["branch", "--list", branch_name])
        if result.stdout.strip():
            return False, f"Branch {branch_name} already exists"

        # Create and checkout new branch
        result = self._run_git(["checkout", "-b", branch_name])
        if result.returncode != 0:
            return False, result.stderr

        return True, branch_name

    def _commit_changes(self, message: str) -> Tuple[bool, str]:
        """Commit staged changes with the given message."""
        result = self._run_git(["commit", "-m", message])
        if result.returncode != 0:
            return False, result.stderr

        return True, result.stdout

    def _push_branch(self, branch_name: str) -> Tuple[bool, str]:
        """Push branch to origin."""
        result = self._run_git(["push", "-u", "origin", branch_name])
        if result.returncode != 0:
            return False, result.stderr

        return True, result.stdout

    def _create_github_pr(
        self,
        title: str,
        body: str,
        base: str = "main",
        draft: bool = False
    ) -> Tuple[bool, str]:
        """Create a PR using GitHub CLI."""
        cmd = [
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--base", base
        ]

        if draft:
            cmd.append("--draft")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return False, result.stderr

            # gh pr create outputs the PR URL on success
            return True, result.stdout.strip()

        except subprocess.TimeoutExpired:
            return False, "PR creation timed out"
        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, str(e)

    def _run_git(self, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        return subprocess.run(
            ["git"] + args,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=timeout
        )

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        # Lowercase and replace spaces/special chars with hyphens
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug

    def get_modified_files(self) -> Tuple[bool, List[str]]:
        """
        Get list of files with staged changes.

        Returns:
            Tuple of (success, list_of_files)
        """
        try:
            result = self._run_git(["diff", "--cached", "--name-only"])
            if result.returncode != 0:
                return False, []

            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            return True, files

        except (OSError, subprocess.SubprocessError):
            return False, []

    def check_gh_auth(self) -> Tuple[bool, str]:
        """
        Check if gh CLI is authenticated.

        Returns:
            Tuple of (authenticated, message)
        """
        if not self._gh_available:
            return False, "GitHub CLI (gh) not installed"

        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, "GitHub CLI authenticated"
            else:
                return False, f"Not authenticated: {result.stderr}\nRun: gh auth login"

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Auth check failed: {str(e)}"

    # Helper methods

    @staticmethod
    def _escape_commit_message(message: str) -> str:
        """Escape commit message for shell."""
        # Replace single quotes with escaped quotes
        return message.replace("'", "'\\''")
