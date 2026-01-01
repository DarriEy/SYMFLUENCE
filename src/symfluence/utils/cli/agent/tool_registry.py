"""
Tool registry for the SYMFLUENCE AI agent.

This module defines all available tools as OpenAI function calling schemas.
Each CLI command is mapped to a function definition that the LLM can call.
"""

from typing import List, Dict, Any


class ToolRegistry:
    """
    Registry of all available tools for the agent.

    Provides tool definitions in OpenAI function calling format and maps
    tool names to their execution functions.
    """

    def __init__(self, cli_manager):
        """
        Initialize the tool registry.

        Args:
            cli_manager: Instance of CLIArgumentManager
        """
        self.cli_manager = cli_manager
        self.tools = self._build_tool_definitions()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions for function calling.

        Returns:
            List of tool definitions in OpenAI format
        """
        return self.tools

    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Build all tool definitions from CLI manager.

        Returns:
            List of tool definitions
        """
        tools = []

        # Add workflow step tools
        tools.extend(self._build_workflow_step_tools())

        # Add binary management tools
        tools.extend(self._build_binary_management_tools())

        # Add configuration tools
        tools.extend(self._build_configuration_tools())

        # Add workflow management tools
        tools.extend(self._build_workflow_management_tools())

        # Add pour point setup tool
        tools.extend(self._build_pour_point_tools())

        # Add meta tools
        tools.extend(self._build_meta_tools())

        return tools

    def _build_workflow_step_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for all workflow steps."""
        tools = []

        for step_name, step_info in self.cli_manager.workflow_steps.items():
            tool = {
                "type": "function",
                "function": {
                    "name": step_name,
                    "description": step_info['description'] + ". Requires a valid SYMFLUENCE configuration file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_path": {
                                "type": "string",
                                "description": "Path to the SYMFLUENCE YAML configuration file"
                            },
                            "debug": {
                                "type": "boolean",
                                "description": "Enable debug output (optional, default: false)"
                            }
                        },
                        "required": ["config_path"]
                    }
                }
            }
            tools.append(tool)

        return tools

    def _build_binary_management_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for binary/executable management."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "install_executables",
                    "description": "Install external modeling tool repositories (summa, mizuroute, fuse, taudem, gistool, datatool, ngen, ngiab). "
                                   "Can install specific tools or all tools if none specified.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tools": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["summa", "mizuroute", "fuse", "taudem", "gistool", "datatool", "ngen", "ngiab", "sundials", "troute"]
                                },
                                "description": "List of specific tools to install. Empty array or omit to install all tools."
                            },
                            "force_install": {
                                "type": "boolean",
                                "description": "Force reinstallation even if tools already exist (optional, default: false)"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_binaries",
                    "description": "Validate that all external tool binaries exist and are functional. "
                                   "Checks for SUMMA, mizuRoute, FUSE, TauDEM, and other installed tools.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_doctor",
                    "description": "Run comprehensive system diagnostics to check binaries, toolchain, and system libraries. "
                                   "Identifies missing dependencies and configuration issues.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_tools_info",
                    "description": "Display information about installed tools from toolchain metadata, "
                                   "including versions, paths, and installation status.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def _build_configuration_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for configuration management."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_config_templates",
                    "description": "List all available SYMFLUENCE configuration templates. "
                                   "Shows template names, descriptions, and file paths.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_config",
                    "description": "Update an existing SYMFLUENCE configuration file with new settings. "
                                   "Prompts for values and preserves existing configuration structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_file": {
                                "type": "string",
                                "description": "Path to the configuration file to update"
                            }
                        },
                        "required": ["config_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_environment",
                    "description": "Validate the system environment and check for required dependencies. "
                                   "Verifies Python version, required packages, and system tools.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_config_file",
                    "description": "Validate a SYMFLUENCE configuration file for correctness. "
                                   "Checks YAML syntax, required fields, and value validity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_file": {
                                "type": "string",
                                "description": "Path to the configuration file to validate"
                            }
                        },
                        "required": ["config_file"]
                    }
                }
            }
        ]

    def _build_workflow_management_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for workflow management."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "show_workflow_status",
                    "description": "Show the current status of a SYMFLUENCE workflow. "
                                   "Displays completed steps, pending steps, and next recommended actions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_path": {
                                "type": "string",
                                "description": "Path to the SYMFLUENCE configuration file"
                            }
                        },
                        "required": ["config_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_workflow_steps",
                    "description": "List all available workflow steps with their descriptions. "
                                   "Shows the complete workflow sequence and what each step does.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "resume_from_step",
                    "description": "Resume a workflow from a specific step onwards. "
                                   "Executes the specified step and all subsequent steps in the workflow.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_path": {
                                "type": "string",
                                "description": "Path to the SYMFLUENCE configuration file"
                            },
                            "step_name": {
                                "type": "string",
                                "description": "Name of the step to resume from"
                            }
                        },
                        "required": ["config_path", "step_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clean_workflow_files",
                    "description": "Clean intermediate or output files from a workflow. "
                                   "Helps reclaim disk space or prepare for re-running steps.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_path": {
                                "type": "string",
                                "description": "Path to the SYMFLUENCE configuration file"
                            },
                            "clean_level": {
                                "type": "string",
                                "enum": ["intermediate", "outputs", "all"],
                                "description": "Level of cleaning: intermediate (temp files), outputs (results), or all"
                            },
                            "dry_run": {
                                "type": "boolean",
                                "description": "Show what would be cleaned without actually deleting (optional, default: false)"
                            }
                        },
                        "required": ["config_path"]
                    }
                }
            }
        ]

    def _build_pour_point_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for pour point workflow setup."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "setup_pour_point_workflow",
                    "description": "Set up a complete SYMFLUENCE workflow for a watershed based on a pour point location. "
                                   "Creates configuration file, defines domain boundaries, and prepares for modeling. "
                                   "This is the recommended way to start a new watershed modeling project.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "Latitude of the pour point in decimal degrees (e.g., 51.1722 for Banff)"
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude of the pour point in decimal degrees (e.g., -115.5717 for Banff)"
                            },
                            "domain_name": {
                                "type": "string",
                                "description": "Name for the watershed/domain (e.g., 'BowAtBanff', 'FyrisRiver')"
                            },
                            "domain_definition_method": {
                                "type": "string",
                                "enum": ["lumped", "point", "subset", "delineate"],
                                "description": "Method for defining domain boundaries: "
                                               "'delineate' - trace watershed boundary from pour point (recommended for watersheds), "
                                               "'lumped' - single modeling unit, "
                                               "'point' - point-scale modeling, "
                                               "'subset' - use custom bounding box"
                            },
                            "bounding_box": {
                                "type": "object",
                                "properties": {
                                    "lat_max": {
                                        "type": "number",
                                        "description": "Maximum latitude (northern bound)"
                                    },
                                    "lon_min": {
                                        "type": "number",
                                        "description": "Minimum longitude (western bound)"
                                    },
                                    "lat_min": {
                                        "type": "number",
                                        "description": "Minimum latitude (southern bound)"
                                    },
                                    "lon_max": {
                                        "type": "number",
                                        "description": "Maximum longitude (eastern bound)"
                                    }
                                },
                                "description": "Optional bounding box for 'subset' domain method. "
                                               "If not provided with delineate method, uses 1-degree buffer around pour point."
                            }
                        },
                        "required": ["latitude", "longitude", "domain_name", "domain_definition_method"]
                    }
                }
            }
        ]

    def _build_meta_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for meta operations (help, info, etc.)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "show_help",
                    "description": "Show help information about agent commands and usage. "
                                   "Provides guidance on how to use the agent and available commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_tools",
                    "description": "List all available tools and their descriptions. "
                                   "Shows the complete set of operations the agent can perform.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "explain_workflow",
                    "description": "Explain the SYMFLUENCE workflow process and step sequence. "
                                   "Provides an overview of how the modeling workflow operates.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
