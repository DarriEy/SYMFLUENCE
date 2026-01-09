import os
import shutil
import subprocess
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .external_tools_config import get_external_tools_definitions

class BinaryManager:
    """
    Manages external tool installation, validation, and execution.
    """
    
    def __init__(self, external_tools: Optional[Dict[str, Any]] = None):
        """
        Initialize the BinaryManager. 
        
        Args:
            external_tools: Dictionary of tool definitions. If None, loads from config.
        """
        self.external_tools = external_tools or get_external_tools_definitions()

    def get_executables(self, specific_tools: List[str] = None, symfluence_instance=None,
                        force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clone and install external tool repositories with dependency resolution.
        """
        print(f"\n🚀 {'Planning' if dry_run else 'Installing'} External Tools:")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN - No actual installation will occur")
            print("-" * 30)
        
        installation_results = {
            'successful': [],
            'failed': [],
            'skipped': [],
            'errors': [],
            'dry_run': dry_run
        }
        
        # Get config
        config = {}
        if symfluence_instance and hasattr(symfluence_instance, 'config'):
            config = symfluence_instance.config
        else:
            try:
                # Try to load template from package data
                from symfluence.resources import get_config_template
                config_path = get_config_template()
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                config = self._ensure_valid_config_paths(config, config_path)
            except:
                pass

        # Determine installation directory
        # Priority: 1) Environment variable, 2) Config file, 3) Current directory
        import os
        data_dir = os.getenv('SYMFLUENCE_DATA') or config.get('SYMFLUENCE_DATA_DIR', '.')
        install_base_dir = Path(data_dir) / 'installs'
        
        print(f"📁 Installation directory: {install_base_dir}")
        
        if not dry_run:
            install_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which tools to install
        if specific_tools is None:
            tools_to_install = list(self.external_tools.keys())
        else:
            tools_to_install = []
            for tool in specific_tools:
                if tool in self.external_tools:
                    tools_to_install.append(tool)
                else:
                    print(f"⚠️  Unknown tool: {tool}")
                    installation_results['errors'].append(f"Unknown tool: {tool}")
        
        # Resolve dependencies and sort by install order
        tools_to_install = self._resolve_tool_dependencies(tools_to_install)
        
        print(f"🎯 Installing tools in order: {', '.join(tools_to_install)}")
        
        # Install each tool
        for tool_name in tools_to_install:
            tool_info = self.external_tools[tool_name]
            print(f"\n🔧 {'Planning' if dry_run else 'Installing'} {tool_name.upper()}:")
            print(f"   📝 {tool_info['description']}")
            
            tool_install_dir = install_base_dir / tool_info['install_dir']
            repository_url = tool_info['repository']
            branch = tool_info.get('branch')
            
            try:
                # Check if already exists
                if tool_install_dir.exists() and not force:
                    print(f"   ⏭️  Skipping - already exists at: {tool_install_dir}")
                    print(f"   💡 Use --force_install to reinstall")
                    installation_results['skipped'].append(tool_name)
                    continue
                
                if dry_run:
                    print(f"   🔍 Would clone: {repository_url}")
                    if branch:
                        print(f"   🌿 Would checkout branch: {branch}")
                    print(f"   📂 Target directory: {tool_install_dir}")
                    print(f"   🔨 Would run build commands:")
                    for cmd in tool_info['build_commands']:
                        print(f"      {cmd[:100]}...")
                    installation_results['successful'].append(f"{tool_name} (dry run)")
                    continue
                
                # Remove existing if force reinstall
                if tool_install_dir.exists() and force:
                    print(f"   🗑️  Removing existing installation: {tool_install_dir}")
                    shutil.rmtree(tool_install_dir)
                
                # Clone repository or create directory for non-git tools
                if repository_url:
                    print(f"   📥 Cloning from: {repository_url}")
                    if branch:
                        print(f"   🌿 Checking out branch: {branch}")
                        clone_cmd = ['git', 'clone', '-b', branch, repository_url, str(tool_install_dir)]
                    else:
                        clone_cmd = ['git', 'clone', repository_url, str(tool_install_dir)]
                    
                    clone_result = subprocess.run(
                        clone_cmd,
                        capture_output=True, text=True, check=True
                    )
                    print(f"   ✅ Clone successful")
                else:
                    print(f"   📂 Creating installation directory")
                    tool_install_dir.mkdir(parents=True, exist_ok=True)
                    print(f"   ✅ Directory created: {tool_install_dir}")
                
                # Check dependencies
                missing_deps = self._check_dependencies(tool_info.get('dependencies', []))
                if missing_deps:
                    print(f"   ⚠️  Missing system dependencies: {', '.join(missing_deps)}")
                    print(f"   💡 These may be available as modules - check with 'module avail'")
                    installation_results['errors'].append(f"{tool_name}: missing system dependencies {missing_deps}")
                
                # Check if tool dependencies (requires) are satisfied
                if 'requires' in tool_info:
                    required_tools = tool_info['requires']
                    for req_tool in required_tools:
                        req_tool_dir = install_base_dir / self.external_tools[req_tool]['install_dir']
                        if not req_tool_dir.exists():
                            error_msg = f"{tool_name} requires {req_tool} but it's not installed"
                            print(f"   ❌ {error_msg}")
                            installation_results['errors'].append(error_msg)
                            installation_results['failed'].append(tool_name)
                            continue
                
                # Run build commands
                if tool_info.get('build_commands'):
                    print(f"   🔨 Running build commands...")
                    
                    original_dir = os.getcwd()
                    os.chdir(tool_install_dir)
                    
                    try:
                        # Combine all commands into a single script to preserve env vars
                        combined_script = "\n".join(tool_info['build_commands'])
                        
                        build_result = subprocess.run(
                            combined_script,
                            shell=True,
                            check=True,
                            capture_output=True,
                            text=True,
                            executable='/bin/bash'
                        )
                        
                        # Show output for critical tools
                        if tool_name in ['summa', 'sundials', 'mizuroute', 'fuse', 'ngen']:
                            if build_result.stdout:
                                print(f"      === Build Output ===")
                                for line in build_result.stdout.strip().split('\n'):
                                    print(f"         {line}")
                        else:
                            if build_result.stdout:
                                lines = build_result.stdout.strip().split('\n')
                                for line in lines[-10:]:
                                    print(f"         {line}")
                        
                        print(f"   ✅ Build successful")
                        installation_results['successful'].append(tool_name)
                    
                    except subprocess.CalledProcessError as build_error:
                        print(f"   ❌ Build failed: {build_error}")
                        if build_error.stdout:
                            print(f"      === Build Output ===")
                            for line in build_error.stdout.strip().split('\n'):
                                print(f"         {line}")
                        if build_error.stderr:
                            print(f"      === Error Output ===")
                            for line in build_error.stderr.strip().split('\n'):
                                print(f"         {line}")
                        installation_results['failed'].append(tool_name)
                        installation_results['errors'].append(f"{tool_name} build failed")
                    
                    finally:
                        os.chdir(original_dir)
                else:
                    print(f"   ✅ No build required")
                    installation_results['successful'].append(tool_name)
                
                # Verify installation
                self._verify_installation(tool_name, tool_info, tool_install_dir)
            
            except subprocess.CalledProcessError as e:
                if repository_url:
                    error_msg = f"Failed to clone {repository_url}: {e.stderr if e.stderr else str(e)}"
                else:
                    error_msg = f"Failed during installation: {e.stderr if e.stderr else str(e)}"
                print(f"   ❌ {error_msg}")
                installation_results['failed'].append(tool_name)
                installation_results['errors'].append(f"{tool_name}: {error_msg}")
            
            except Exception as e:
                error_msg = f"Installation error: {str(e)}"
                print(f"   ❌ {error_msg}")
                installation_results['failed'].append(tool_name)
                installation_results['errors'].append(f"{tool_name}: {error_msg}")
        
        # Print summary
        self._print_installation_summary(installation_results, dry_run)
        
        return installation_results

    def validate_binaries(self, symfluence_instance=None) -> Any:
        """
        Validate that required binary executables exist and are functional.
        """
        print("\n🔧 Validating External Tool Binaries:")
        print("=" * 50)

        validation_results = {
            'valid_tools': [],
            'missing_tools': [],
            'failed_tools': [],
            'warnings': [],
            'summary': {}
        }

        # Get config if available
        config = {}
        if symfluence_instance and hasattr(symfluence_instance, 'config'):
            config = symfluence_instance.config
        elif symfluence_instance and hasattr(symfluence_instance, 'workflow_orchestrator'):
            config = symfluence_instance.workflow_orchestrator.config

        # If no config available, try to load default from package data
        if not config:
            try:
                from symfluence.resources import get_config_template
                config_path = get_config_template()
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"📄 Using config template from package: {config_path}")
            except FileNotFoundError:
                print("⚠️  No configuration file found - using default paths")
            except Exception as e:
                print(f"⚠️  Could not load config: {str(e)} - using default paths")

        # Validate each tool
        for tool_name, tool_info in self.external_tools.items():
            print(f"\n🔍 Checking {tool_name.upper()}:")
            tool_result = {
                'name': tool_name,
                'description': tool_info['description'],
                'status': 'unknown',
                'path': None,
                'executable': None,
                'version': None,
                'errors': []
            }

            try:
                # Determine tool path (config override or default)
                config_path_key = tool_info['config_path_key']
                tool_path = config.get(config_path_key, 'default')
                if tool_path == 'default':
                    data_dir = config.get('SYMFLUENCE_DATA_DIR', '.')
                    tool_path = Path(data_dir) / tool_info['default_path_suffix']
                else:
                    tool_path = Path(tool_path)
                tool_result['path'] = str(tool_path)

                # If tool defines a verify_install block, honor it first
                verify = tool_info.get('verify_install')
                found_path = None
                if verify and isinstance(verify, dict):
                    check_type = verify.get('check_type', 'exists_all')
                    candidates = [tool_path / p for p in verify.get('file_paths', [])]

                    if check_type == 'exists_any':
                        for p in candidates:
                            if p.exists():
                                found_path = p
                                break
                        exists_ok = found_path is not None
                    elif check_type in ('exists_all', 'exists'):
                        exists_ok = all(p.exists() for p in candidates)
                        if exists_ok:
                            found_path = next((p for p in candidates if p.exists()), candidates[0] if candidates else tool_path)
                    else:
                        exists_ok = False

                    if exists_ok:
                        test_cmd = tool_info.get('test_command')
                        if test_cmd is None:
                            tool_result['status'] = 'valid'
                            tool_result['version'] = 'Installed (existence verified)'
                            validation_results['valid_tools'].append(tool_name)
                            print(f"   ✅ Found at: {found_path}")
                            print(f"   ✅ Status: Installed")
                            validation_results['summary'][tool_name] = tool_result
                            continue

                # Fallback: single-executable check
                config_exe_key = tool_info.get('config_exe_key')
                if config_exe_key and config_exe_key in config:
                    exe_name = config[config_exe_key]
                else:
                    exe_name = tool_info['default_exe']
                tool_result['executable'] = exe_name

                exe_path = tool_path / exe_name
                if not exe_path.exists():
                    exe_name_no_ext = exe_name.replace('.exe', '')
                    exe_path_no_ext = tool_path / exe_name_no_ext
                    if exe_path_no_ext.exists():
                        exe_path = exe_path_no_ext
                        tool_result['executable'] = exe_name_no_ext

                if exe_path.exists():
                    test_cmd = tool_info.get('test_command')
                    if test_cmd is None:
                        tool_result['status'] = 'valid'
                        tool_result['version'] = 'Installed (existence verified)'
                        validation_results['valid_tools'].append(tool_name)
                        print(f"   ✅ Found at: {exe_path}")
                        print(f"   ✅ Status: Installed")
                    else:
                        try:
                            result = subprocess.run(
                                [str(exe_path), test_cmd],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if result.returncode == 0 or test_cmd == '--help' or tool_name in ('taudem',):
                                tool_result['status'] = 'valid'
                                tool_result['version'] = (result.stdout.strip()[:100] if result.stdout else 'Available')
                                validation_results['valid_tools'].append(tool_name)
                                print(f"   ✅ Found at: {exe_path}")
                                print(f"   ✅ Status: Working")
                            else:
                                tool_result['status'] = 'failed'
                                tool_result['errors'].append(f"Test command failed: {result.stderr}")
                                validation_results['failed_tools'].append(tool_name)
                                print(f"   🟡 Found but test failed: {exe_path}")
                                print(f"   ⚠️  Error: {result.stderr[:100]}")

                        except subprocess.TimeoutExpired:
                            tool_result['status'] = 'timeout'
                            tool_result['errors'].append("Test command timed out")
                            validation_results['warnings'].append(f"{tool_name}: test timed out")
                            print(f"   🟡 Found but test timed out: {exe_path}")
                        except Exception as test_error:
                            tool_result['status'] = 'test_error'
                            tool_result['errors'].append(f"Test error: {str(test_error)}")
                            validation_results['warnings'].append(f"{tool_name}: {str(test_error)}")
                            print(f"   🟡 Found but couldn't test: {exe_path}")
                            print(f"   ⚠️  Test error: {str(test_error)}")

                else:
                    tool_result['status'] = 'missing'
                    tool_result['errors'].append(f"Executable not found at: {exe_path}")
                    validation_results['missing_tools'].append(tool_name)
                    print(f"   ❌ Not found: {exe_path}")
                    print(f"   💡 Try: python SYMFLUENCE.py --get_executables {tool_name}")

            except Exception as e:
                tool_result['status'] = 'error'
                tool_result['errors'].append(f"Validation error: {str(e)}")
                validation_results['failed_tools'].append(tool_name)
                print(f"   ❌ Validation error: {str(e)}")

            validation_results['summary'][tool_name] = tool_result

        # Print summary
        total_tools = len(self.external_tools)
        valid_count = len(validation_results['valid_tools'])
        missing_count = len(validation_results['missing_tools'])
        failed_count = len(validation_results['failed_tools'])

        print(f"\n📊 Binary Validation Summary:")
        print(f"   ✅ Valid: {valid_count}/{total_tools}")
        print(f"   ❌ Missing: {missing_count}/{total_tools}")
        print(f"   🔧 Failed: {failed_count}/{total_tools}")

        if len(validation_results['missing_tools']) == 0 and len(validation_results['failed_tools']) == 0:
            return True
        else:
            return validation_results

    def handle_binary_management(self, execution_plan: Dict[str, Any]) -> bool:
        """
        Legacy dispatcher for binary management operations.
        """
        ops = execution_plan.get('binary_operations', {})
        
        if ops.get('doctor'):
            self.run_doctor()
            return True
        if ops.get('tools_info'):
            self.show_tools_info()
            return True
        if ops.get('validate_binaries'):
            return self.validate_binaries() is True
        if ops.get('get_executables'):
            tools = ops.get('get_executables')
            if isinstance(tools, bool):
                tools = None
            result = self.get_executables(specific_tools=tools)
            return len(result.get('failed', [])) == 0
            
        return False

    def run_doctor(self) -> bool:
        """Alias for doctor() for backward compatibility."""
        return self.doctor()

    def show_tools_info(self) -> bool:
        """Alias for tools_info() for backward compatibility."""
        return self.tools_info()

    def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check which dependencies are missing from the system."""
        missing_deps = []
        for dep in dependencies:
            if not shutil.which(dep):
                missing_deps.append(dep)
        return missing_deps

    def _verify_installation(self, tool_name: str, tool_info: Dict[str, Any],
                            install_dir: Path) -> bool:
        """Verify that a tool was installed correctly."""
        try:
            verify = tool_info.get('verify_install')
            if verify and isinstance(verify, dict):
                check_type = verify.get('check_type', 'exists_all')
                candidates = [install_dir / p for p in verify.get('file_paths', [])]

                if check_type == 'exists_any':
                    ok = any(p.exists() for p in candidates)
                elif check_type in ('exists_all', 'exists'):
                    ok = all(p.exists() for p in candidates)
                else:
                    ok = False

                print(f"   {'✅' if ok else '❌'} Install verification ({check_type}):")
                for p in candidates:
                    print(f"      {'✓' if p.exists() else '✗'} {p}")
                return ok

            exe_name = tool_info.get('default_exe')
            if not exe_name:
                return False

            possible_paths = [
                install_dir / exe_name,
                install_dir / 'bin' / exe_name,
                install_dir / 'build' / exe_name,
                install_dir / 'route' / 'bin' / exe_name,
                install_dir / exe_name.replace('.exe', ''),
                install_dir / 'install' / 'sundials' / exe_name,
            ]

            for exe_path in possible_paths:
                if exe_path.exists():
                    print(f"   ✅ Executable/library found: {exe_path}")
                    return True

            return False

        except Exception as e:
            print(f"   ⚠️  Verification error: {str(e)}")
            return False

    def _resolve_tool_dependencies(self, tools: List[str]) -> List[str]:
        """Resolve dependencies between tools and return sorted list."""
        tools_with_deps = set(tools)
        for tool in tools:
            if tool in self.external_tools and 'requires' in self.external_tools[tool]:
                required = self.external_tools[tool]['requires']
                tools_with_deps.update(required)
        
        return sorted(
            tools_with_deps,
            key=lambda t: (self.external_tools.get(t, {}).get('order', 999), t)
        )

    def _print_installation_summary(self, results: Dict[str, Any], dry_run: bool) -> None:
        """Print installation summary."""
        successful_count = len(results['successful'])
        failed_count = len(results['failed'])
        skipped_count = len(results['skipped'])
        
        print(f"\n📊 Installation Summary:")
        if dry_run:
            print(f"   🔍 Would install: {successful_count} tools")
            print(f"   ⏭️  Would skip: {skipped_count} tools")
        else:
            print(f"   ✅ Successful: {successful_count} tools")
            print(f"   ❌ Failed: {failed_count} tools")
            print(f"   ⏭️  Skipped: {skipped_count} tools")
        
        if results['errors']:
            print(f"\n❌ Errors encountered:")
            for error in results['errors']:
                print(f"   • {error}")

    def _ensure_valid_config_paths(self, config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
        """
        Ensure SYMFLUENCE_DATA_DIR and SYMFLUENCE_CODE_DIR paths exist and are valid.
        """
        data_dir = config.get('SYMFLUENCE_DATA_DIR')
        code_dir = config.get('SYMFLUENCE_CODE_DIR')

        data_dir_valid = False
        code_dir_valid = False

        if data_dir:
            try:
                data_path = Path(data_dir)
                if data_path.exists():
                    test_file = data_path / '.symfluence_test'
                    try:
                        test_file.touch()
                        test_file.unlink()
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
                else:
                    try:
                        data_path.mkdir(parents=True, exist_ok=True)
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
            except Exception:
                pass

        if code_dir:
            try:
                code_path = Path(code_dir)
                if code_path.exists() and os.access(code_path, os.R_OK):
                    code_dir_valid = True
            except Exception:
                pass

        if not data_dir_valid or not code_dir_valid:
            print(f"\n⚠️  Detected invalid or inaccessible paths in config template:")

            if not code_dir_valid:
                new_code_dir = Path.cwd().resolve()
                config['SYMFLUENCE_CODE_DIR'] = str(new_code_dir)
                print(f"   ✅ SYMFLUENCE_CODE_DIR set to: {new_code_dir}")

            if not data_dir_valid:
                new_data_dir = (Path.cwd().parent / 'SYMFLUENCE_data').resolve()
                config['SYMFLUENCE_DATA_DIR'] = str(new_data_dir)
                try:
                    new_data_dir.mkdir(parents=True, exist_ok=True)
                    print(f"   ✅ SYMFLUENCE_DATA_DIR set to: {new_data_dir}")
                except Exception:
                    pass

            try:
                backup_path = config_path.with_name(f"{config_path.stem}_backup{config_path.suffix}")
                if config_path.exists():
                    shutil.copy2(config_path, backup_path)
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception:
                pass

        return config

    def detect_npm_binaries(self) -> Optional[Path]:
        """
        Detect if SYMFLUENCE binaries are installed via npm.

        Returns:
            Path to npm-installed binaries, or None if not found
        """
        try:
            # Try to get global npm root
            result = subprocess.run(
                ['npm', 'root', '-g'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                npm_root = Path(result.stdout.strip())
                npm_bin_dir = npm_root / 'symfluence' / 'dist' / 'bin'

                if npm_bin_dir.exists() and npm_bin_dir.is_dir():
                    return npm_bin_dir

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return None

    def doctor(self) -> bool:
        """
        Run system diagnostics: check binaries, toolchain, and system libraries.
        """
        print("\n" + "=" * 60)

        # Check binaries
        print("\n📦 Checking binaries...")
        print("-" * 40)

        symfluence_data = os.getenv('SYMFLUENCE_DATA')
        npm_bin_dir = self.detect_npm_binaries()

        if npm_bin_dir:
            print(f"   ℹ️  Detected npm-installed binaries: {npm_bin_dir}")

        found_binaries = 0
        total_binaries = 0

        # Use external_tools definitions for more accurate checking
        for name, tool_info in self.external_tools.items():
            if name == 'sundials': continue # Skip library-only tool
            total_binaries += 1
            
            found = False
            location = None
            
            # 1. Check in SYMFLUENCE_DATA (installed from source)
            if symfluence_data:
                rel_path_suffix = tool_info.get('default_path_suffix', '')
                exe_name = tool_info.get('default_exe', '')
                
                # Check for folder (e.g. TauDEM) or specific file
                full_path = Path(symfluence_data) / rel_path_suffix
                if name in ('taudem', 'wmfire'):
                    if full_path.exists() and full_path.is_dir():
                        found = True
                        location = full_path
                elif exe_name:
                    exe_path = full_path / exe_name
                    if exe_path.exists():
                        found = True
                        location = exe_path
                    else:
                        # Try without extension
                        exe_path_no_ext = full_path / exe_name.replace('.exe', '')
                        if exe_path_no_ext.exists():
                            found = True
                            location = exe_path_no_ext

            # 2. Check npm installation as fallback
            if not found and npm_bin_dir:
                # Check for the tool name directly (e.g., 'summa')
                npm_path = npm_bin_dir / name
                if npm_path.exists():
                    found = True
                    location = npm_path
                else:
                    # Check for the default executable name (e.g., 'pitremove' for TauDEM)
                    exe_name = tool_info.get('default_exe', '')
                    if exe_name:
                        # Try both with and without .exe
                        for candidate in [exe_name, exe_name.replace('.exe', '')]:
                            npm_exe_path = npm_bin_dir / candidate
                            if npm_exe_path.exists():
                                found = True
                                location = npm_exe_path
                                break

            if found:
                print(f"   ✅ {name:<12} {location}")
                found_binaries += 1
            else:
                print(f"   ❌ {name:<12} Not found")

        # Check toolchain metadata
        print("\n🔧 Toolchain metadata...")
        print("-" * 40)

        toolchain_locations = []
        if symfluence_data:
            toolchain_locations.append(Path(symfluence_data) / 'installs' / 'toolchain.json')
        if npm_bin_dir:
            toolchain_locations.append(npm_bin_dir.parent / 'toolchain.json')

        toolchain_found = False
        for toolchain_path in toolchain_locations:
            if toolchain_path.exists():
                try:
                    import json
                    with open(toolchain_path) as f:
                        toolchain = json.load(f)

                    platform = toolchain.get('platform', 'unknown')
                    build_date = toolchain.get('build_date', 'unknown')
                    fortran = toolchain.get('compilers', {}).get('fortran', 'unknown')

                    print(f"   ✅ Found: {toolchain_path}")
                    print(f"      Platform: {platform}")
                    print(f"      Build date: {build_date}")
                    print(f"      Fortran: {fortran}")
                    toolchain_found = True
                    break
                except Exception as e:
                    print(f"   ⚠️  Error reading {toolchain_path}: {e}")

        if not toolchain_found:
            print("   ❌ No toolchain metadata found")

        # Check system libraries
        print("\n📚 System libraries...")
        print("-" * 40)

        system_tools = {
            'nc-config': 'NetCDF',
            'nf-config': 'NetCDF-Fortran',
            'h5cc': 'HDF5',
            'gdal-config': 'GDAL',
            'mpirun': 'MPI',
        }

        found_libs = 0
        for tool, name in system_tools.items():
            location = shutil.which(tool)
            if location:
                print(f"   ✅ {name:<16} {location}")
                found_libs += 1
            else:
                print(f"   ❌ {name:<16} Not found")

        # Summary
        print("\n" + "=" * 60)
        print("📊 Summary:")
        print(f"   Binaries: {found_binaries}/{total_binaries} found")
        print(f"   Toolchain metadata: {'✅ Found' if toolchain_found else '❌ Not found'}")
        print(f"   System libraries: {found_libs}/{len(system_tools)} found")

        if found_binaries == total_binaries and toolchain_found and found_libs >= 3:
            print("\n✅ System is ready for SYMFLUENCE!")
        elif found_binaries == 0:
            print("\n⚠️  No binaries found. Install with:")
            print("   • npm install -g symfluence (for pre-built binaries)")
            print("   • ./symfluence --get_executables (to build from source)")
        else:
            print("\n⚠️  Some components missing. Review output above.")

        print("=" * 60 + "\n")
        return True

    def tools_info(self) -> bool:
        """
        Display installed tools information from toolchain metadata.
        """
        symfluence_data = os.getenv('SYMFLUENCE_DATA')
        npm_bin_dir = self.detect_npm_binaries()

        toolchain_locations = []
        if symfluence_data:
            toolchain_locations.append(Path(symfluence_data) / 'installs' / 'toolchain.json')
        if npm_bin_dir:
            toolchain_locations.append(npm_bin_dir.parent / 'toolchain.json')

        toolchain_path = None
        for path in toolchain_locations:
            if path.exists():
                toolchain_path = path
                break

        if not toolchain_path:
            print("❌ No toolchain metadata found.")
            print("\nℹ️  Toolchain metadata is generated during installation.")
            print("   Install binaries with:")
            print("   • npm install -g symfluence")
            print("   • ./symfluence --get_executables\n")
            return

        try:
            import json
            with open(toolchain_path) as f:
                toolchain = json.load(f)

            print("=" * 60)
            print(f"Platform: {toolchain.get('platform', 'unknown')}")
            print(f"Build Date: {toolchain.get('build_date', 'unknown')}")
            print(f"Toolchain file: {toolchain_path}")

            # Compilers
            if 'compilers' in toolchain:
                print("\n🛠️  Compilers:")
                print("-" * 40)
                compilers = toolchain['compilers']
                for key, value in compilers.items():
                    print(f"   {key.capitalize():<12} {value}")

            # Libraries
            if 'libraries' in toolchain:
                print("\n📚 Libraries:")
                print("-" * 40)
                libraries = toolchain['libraries']
                for key, value in libraries.items():
                    print(f"   {key.capitalize():<12} {value}")

            # Tools
            if 'tools' in toolchain:
                print("\n🔨 Installed Tools:")
                print("-" * 40)
                for tool_name, tool_info in toolchain['tools'].items():
                    print(f"\n   {tool_name.upper()}:")
                    if 'commit' in tool_info:
                        commit_short = tool_info['commit'][:8] if len(tool_info['commit']) > 8 else tool_info['commit']
                        print(f"      Commit: {commit_short}")
                    if 'branch' in tool_info:
                        print(f"      Branch: {tool_info['branch']}")
                    if 'executable' in tool_info:
                        print(f"      Executable: {tool_info['executable']}")

            print("\n" + "=" * 60 + "\n")
            return True

        except json.JSONDecodeError:
            print(f"❌ Error: Toolchain file is not valid JSON: {toolchain_path}")
            return False
        except Exception as e:
            print(f"❌ Error reading toolchain file: {e}")
            return False
