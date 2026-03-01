# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Code analysis module for understanding project structure.

Provides analysis of project layout, file relationships, and code structure.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CodeAnalyzer:
    """Analyze project codebase structure and relationships."""

    def __init__(self, repo_root: Optional[str] = None):
        """
        Initialize code analyzer.

        Args:
            repo_root: Repository root directory. If None, uses current directory.
        """
        if repo_root is None:
            import os
            repo_root = os.getcwd()
        self.repo_root = Path(repo_root).resolve()

    def analyze_project_structure(self, depth: str = "quick") -> Tuple[bool, str]:
        """
        Analyze project structure and provide overview.

        Args:
            depth: Analysis depth - 'quick', 'detailed', or 'deep'

        Returns:
            Tuple of (success, analysis_or_error)
        """
        try:
            # Find key directories
            key_dirs = self._find_key_directories()

            # Count Python files
            py_files = list(self.repo_root.rglob("*.py"))
            py_count = len([f for f in py_files if '.git' not in f.parts])

            # Analyze each key directory
            dir_info = {}
            for dir_name, dir_path in key_dirs.items():
                if dir_path.exists():
                    files = list(dir_path.rglob("*.py"))
                    file_count = len(files)
                    dir_info[dir_name] = {
                        "files": file_count,
                        "path": str(dir_path.relative_to(self.repo_root)),
                        "subdirs": len([d for d in dir_path.iterdir() if d.is_dir()])
                    }

            # Build output
            output = "Project Structure Analysis\n" + "=" * 50 + "\n\n"

            if depth == "quick":
                output += f"Total Python files: {py_count}\n\n"
                output += "Key directories:\n"
                for name, info in dir_info.items():
                    output += f"  • {name}: {info['files']} Python files, {info['subdirs']} subdirectories\n"

            elif depth == "detailed":
                output += f"Total Python files: {py_count}\n\n"
                output += "Directory Breakdown:\n"
                for name, info in dir_info.items():
                    output += f"\n  {name} ({info['path']})\n"
                    output += f"    - Python files: {info['files']}\n"
                    output += f"    - Subdirectories: {info['subdirs']}\n"

                # Analyze imports
                output += "\n" + "-" * 50 + "\n"
                output += "Key Dependencies:\n"
                deps = self._analyze_imports()
                for module, count in sorted(deps.items(), key=lambda x: x[1], reverse=True)[:10]:
                    output += f"  • {module}: {count} imports\n"

            elif depth == "deep":
                output += self._deep_analysis()

            return True, output

        except (FileNotFoundError, PermissionError, OSError) as e:
            return False, f"Error analyzing project: {str(e)}"

    def find_related_files(self, target_file: str) -> Tuple[bool, str]:
        """
        Find files related to a target file (imports and usage).

        Args:
            target_file: Path to target file relative to repo root

        Returns:
            Tuple of (success, related_files_or_error)
        """
        try:
            target_path = (self.repo_root / target_file).resolve()

            if not target_path.exists():
                return False, f"File not found: {target_file}"

            # Find files that import this module
            importers = self._find_importers(target_path)

            # Find files this module imports from src/
            imports = self._find_internal_imports(target_path)

            output = f"Related Files for: {target_file}\n" + "=" * 50 + "\n\n"

            if imports:
                output += f"Internal Dependencies ({len(imports)}):\n"
                for imp in sorted(imports):
                    output += f"  ← {imp}\n"

            if importers:
                output += f"\nFiles that Import This ({len(importers)}):\n"
                for imp in sorted(importers):
                    output += f"  → {imp}\n"

            if not imports and not importers:
                output += "No related files found"

            return True, output

        except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError) as e:
            return False, f"Error finding related files: {str(e)}"

    def validate_python_syntax(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate Python file syntax.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (success, message_or_error)
        """
        try:
            full_path = (self.repo_root / file_path).resolve()

            if not full_path.exists():
                return False, f"File not found: {file_path}"

            if not full_path.suffix == '.py':
                return False, "Not a Python file"

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to parse as AST
            try:
                ast.parse(content)
                return True, f"✓ Syntax valid: {file_path}"
            except SyntaxError as e:
                return False, f"Syntax error on line {e.lineno}: {e.msg}"

        except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError) as e:
            return False, f"Error validating syntax: {str(e)}"

    def get_file_summary(self, file_path: str) -> Tuple[bool, str]:
        """
        Get a quick summary of a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (success, summary_or_error)
        """
        try:
            full_path = (self.repo_root / file_path).resolve()

            if not full_path.exists():
                return False, f"File not found: {file_path}"

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Parse for structure
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return False, "File has syntax errors"

            # Extract docstring
            docstring = ast.get_docstring(tree) or "(no docstring)"

            # Count definitions
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

            output = f"File Summary: {file_path}\n" + "=" * 50 + "\n\n"
            output += f"Lines: {len(lines)}\n"
            output += f"Imports: {len(imports)}\n"
            output += f"Classes: {len(classes)}\n"
            output += f"Functions: {len(functions)}\n\n"

            output += f"Docstring:\n{docstring}\n\n"

            if classes:
                output += f"Classes ({len(classes)}):\n"
                for cls in classes[:10]:  # Show first 10
                    output += f"  • {cls}\n"

            if functions:
                output += f"\nFunctions ({len(functions)}):\n"
                for func in functions[:10]:  # Show first 10
                    output += f"  • {func}\n"

            return True, output

        except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError, SyntaxError) as e:
            return False, f"Error summarizing file: {str(e)}"

    # Helper methods

    def _find_key_directories(self) -> Dict[str, Path]:
        """Find key directories in the project."""
        return {
            "src": self.repo_root / "src",
            "tests": self.repo_root / "tests",
            "agent": self.repo_root / "src" / "symfluence" / "agent",
            "core": self.repo_root / "src" / "symfluence" / "core",
            "models": self.repo_root / "src" / "symfluence" / "models",
        }

    def _analyze_imports(self) -> Dict[str, int]:
        """Analyze import frequencies across codebase."""
        imports: Dict[str, int] = {}

        for py_file in self.repo_root.rglob("*.py"):
            if '.git' in py_file.parts or '__pycache__' in py_file.parts:
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name.split('.')[0]
                            imports[module] = imports.get(module, 0) + 1
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            module = node.module.split('.')[0]
                            imports[module] = imports.get(module, 0) + 1
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue  # Skip unreadable or unparseable files

        return imports

    def _find_importers(self, target_path: Path) -> List[str]:
        """Find files that import the target module."""
        importers = []
        module_name = self._path_to_module(target_path)

        for py_file in self.repo_root.rglob("*.py"):
            if '.git' in py_file.parts or '__pycache__' in py_file.parts:
                continue
            if py_file == target_path:
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if module_name in content:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            if self._node_imports_module(node, module_name):
                                rel_path = py_file.relative_to(self.repo_root)
                                importers.append(str(rel_path))
                                break
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue  # Skip unreadable or unparseable files

        return list(set(importers))

    def _find_internal_imports(self, target_path: Path) -> List[str]:
        """Find internal modules imported by target file."""
        imports = []

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith('symfluence'):
                            imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('symfluence'):
                        imports.append(node.module)
        except (OSError, UnicodeDecodeError, SyntaxError):
            pass  # Skip unreadable or unparseable files

        return list(set(imports))

    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module name."""
        try:
            rel_path = path.relative_to(self.repo_root)
            # Remove src/ prefix if present
            parts = rel_path.parts
            if parts[0] == 'src':
                parts = parts[1:]
            # Remove .py extension
            if parts[-1].endswith('.py'):
                parts = parts[:-1] + (parts[-1][:-3],)
            return '.'.join(parts)
        except (ValueError, IndexError):
            return str(path)

    def _node_imports_module(self, node: Any, module_name: str) -> bool:
        """Check if AST node imports a module."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name or alias.name.startswith(module_name + '.'):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == module_name or node.module.startswith(module_name)):
                return True
        return False

    def _deep_analysis(self) -> str:
        """Perform deep analysis of codebase."""
        output = "Deep Project Analysis\n" + "=" * 50 + "\n\n"

        # Find largest files
        all_files = list(self.repo_root.rglob("*.py"))
        all_files = [f for f in all_files if '.git' not in f.parts]
        all_files.sort(key=lambda x: x.stat().st_size, reverse=True)

        output += "Largest Python Files:\n"
        for f in all_files[:5]:
            size = f.stat().st_size
            lines = len(open(f, encoding='utf-8').readlines())
            output += f"  • {f.relative_to(self.repo_root)}: {size} bytes, {lines} lines\n"

        # Find most imported modules
        output += "\nMost Imported Modules:\n"
        deps = self._analyze_imports()
        for module, count in sorted(deps.items(), key=lambda x: x[1], reverse=True)[:10]:
            output += f"  • {module}: {count} imports\n"

        return output
