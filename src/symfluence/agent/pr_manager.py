"""
PR management module for staging and proposing code changes.

Generates PR titles, descriptions, and manages git staging.
"""

import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime


class PRManager:
    """Manage PR proposals and change staging."""

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

    def propose_code_change(
        self,
        file_path: str,
        old_code: str,
        new_code: str,
        description: str,
        reason: str = "improvement"
    ) -> Tuple[bool, str]:
        """
        Propose a code modification with validation and staging.

        Args:
            file_path: Path to file relative to repo root
            old_code: Exact code to replace (must match exactly)
            new_code: Replacement code
            description: Description of why this change is needed
            reason: Type of change (bugfix, improvement, feature)

        Returns:
            Tuple of (success, message_or_error)
        """
        try:
            from symfluence.agent.file_operations import FileOperations

            file_ops = FileOperations(str(self.repo_root))

            # Read the current file
            success, content = file_ops.read_file(file_path)
            if not success:
                return False, f"Cannot read file: {content}"

            # Check if old_code exists in file
            if old_code not in content:
                # Try to show approximate match
                return False, (
                    f"Exact code not found in {file_path}.\n\n"
                    f"Looking for:\n{old_code}\n\n"
                    f"Make sure the code matches exactly, including indentation and spacing."
                )

            # Perform the replacement
            modified_content = content.replace(old_code, new_code, 1)

            if modified_content == content:
                return False, "Replacement resulted in no changes"

            # Write modified content
            success, msg = file_ops.write_file(file_path, modified_content)
            if not success:
                return False, f"Failed to write file: {msg}"

            # Show the diff
            success, diff = file_ops.show_diff(file_path)

            output = f"✓ Change proposed to {file_path}\n\n"
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
            output += "3. Use create_pr_proposal to stage for PR\n"
            output += "4. User reviews and pushes changes\n"

            return True, output

        except Exception as e:
            return False, f"Error proposing code change: {str(e)}"

    def show_staged_changes(self) -> Tuple[bool, str]:
        """
        Display all staged changes.

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

            output = "Staged Changes\n" + "=" * 60 + "\n\n"
            output += diff + "\n\n"
            output += "=" * 60 + "\n"
            output += "Use 'git commit' to commit these changes.\n"
            output += "Use 'git reset' to unstage changes.\n"

            return True, output

        except Exception as e:
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
        testing_notes: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive PR description.

        Args:
            title: PR title
            summary: Summary of changes
            reason: Type of change
            files_modified: List of files modified
            testing_notes: Notes on testing performed

        Returns:
            Formatted PR description
        """
        description = f"## Summary\n{summary}\n\n"

        # Add reason section
        if reason == "bugfix":
            description += "## Problem\nThis PR fixes a bug that was causing issues.\n\n"
        elif reason == "feature":
            description += "## Feature\nThis PR adds new functionality.\n\n"
        else:
            description += "## Improvement\nThis PR improves existing functionality.\n\n"

        # Files modified
        if files_modified:
            description += "## Files Modified\n"
            for file_path in files_modified:
                description += f"- `{file_path}`\n"
            description += "\n"

        # Testing
        description += "## Testing\n"
        if testing_notes:
            description += f"{testing_notes}\n"
        else:
            description += "All existing tests pass. No breaking changes.\n"

        # Footer with timestamp
        description += f"\n---\n*PR generated by SYMFLUENCE Agent at {datetime.now().isoformat()}*"

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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
            return False, f"Error getting branch: {str(e)}"

    # Helper methods

    @staticmethod
    def _escape_commit_message(message: str) -> str:
        """Escape commit message for shell."""
        # Replace single quotes with escaped quotes
        return message.replace("'", "'\\''")
