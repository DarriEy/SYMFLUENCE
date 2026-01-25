Agent Tools Reference
=====================

The SYMFLUENCE agent provides 30+ tools organized into categories. Each tool
can be invoked via natural language or explicitly by name.

Workflow Step Tools
-------------------

These tools execute individual workflow steps. Each requires a ``config_path``
parameter pointing to a valid SYMFLUENCE YAML configuration file.

setup_project
^^^^^^^^^^^^^
Initialize project directory structure for a new domain.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

acquire_attributes
^^^^^^^^^^^^^^^^^^
Download geospatial attribute data (soil, land cover, topography, etc.).

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

acquire_forcings
^^^^^^^^^^^^^^^^
Download meteorological forcing data for the domain.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

define_domain
^^^^^^^^^^^^^
Define hydrological domain boundaries using configured method.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

discretize_domain
^^^^^^^^^^^^^^^^^
Discretize domain into modeling units (HRUs, GRUs).

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

model_agnostic_preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Perform model-agnostic data preprocessing.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

model_specific_preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Set up model-specific input files and configurations.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

run_model
^^^^^^^^^
Execute the configured hydrological model.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

postprocess_results
^^^^^^^^^^^^^^^^^^^
Analyze and visualize model results.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``debug``: Enable debug output (default: false)

Binary Management Tools
-----------------------

Tools for managing external model binaries and dependencies.

install_executables
^^^^^^^^^^^^^^^^^^^
Install external modeling tools (SUMMA, mizuRoute, FUSE, etc.).

**Parameters:**

- ``tools``: List of tools to install. Options: summa, mizuroute, fuse, taudem,
  gistool, datatool, ngen, ngiab, sundials, troute. Omit to install all.
- ``force_install``: Force reinstall even if already present (default: false)

**Example:**

.. code-block:: text

    Install SUMMA and mizuRoute for me

validate_binaries
^^^^^^^^^^^^^^^^^
Validate that required model binaries exist and are functional.

**Parameters:** None

run_doctor
^^^^^^^^^^
Run comprehensive system diagnostics.

**Parameters:** None

show_tools_info
^^^^^^^^^^^^^^^
Display information about installed tools including versions and paths.

**Parameters:** None

Configuration Tools
-------------------

Tools for managing SYMFLUENCE configuration files.

list_config_templates
^^^^^^^^^^^^^^^^^^^^^
List all available configuration templates.

**Parameters:** None

update_config
^^^^^^^^^^^^^
Update an existing configuration file with new settings.

**Parameters:**

- ``config_file`` (required): Path to configuration file

validate_environment
^^^^^^^^^^^^^^^^^^^^
Validate system environment and dependencies.

**Parameters:** None

validate_config_file
^^^^^^^^^^^^^^^^^^^^
Validate a configuration file for correctness.

**Parameters:**

- ``config_file`` (required): Path to configuration file

Workflow Management Tools
-------------------------

Tools for managing and monitoring workflow execution.

show_workflow_status
^^^^^^^^^^^^^^^^^^^^
Show current status of a workflow including completed and pending steps.

**Parameters:**

- ``config_path`` (required): Path to configuration file

list_workflow_steps
^^^^^^^^^^^^^^^^^^^
List all available workflow steps with descriptions.

**Parameters:** None

resume_from_step
^^^^^^^^^^^^^^^^
Resume a workflow from a specific step onwards.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``step_name`` (required): Name of step to resume from

clean_workflow_files
^^^^^^^^^^^^^^^^^^^^
Clean intermediate or output files from a workflow.

**Parameters:**

- ``config_path`` (required): Path to configuration file
- ``clean_level``: Level of cleaning - intermediate, outputs, or all
- ``dry_run``: Preview what would be cleaned (default: false)

Domain Setup Tools
------------------

Tools for setting up new modeling domains.

setup_pour_point_workflow
^^^^^^^^^^^^^^^^^^^^^^^^^
Set up a complete workflow for a watershed based on a pour point location.

**Parameters:**

- ``latitude`` (required): Latitude in decimal degrees
- ``longitude`` (required): Longitude in decimal degrees
- ``domain_name`` (required): Name for the domain
- ``domain_definition_method`` (required): Method - point, lumped, semidistributed, or distributed
- ``bounding_box``: Optional bounding box coordinates

**Example:**

.. code-block:: text

    Set up a watershed project for coordinates 51.17, -115.57
    named BowAtBanff using delineation

Code Operations Tools
---------------------

Tools for code analysis and modification (agent self-improvement).

read_file
^^^^^^^^^
Read a source code file with line numbers.

**Parameters:**

- ``file_path`` (required): Path relative to repository root
- ``start_line``: Start line for partial read
- ``end_line``: End line for partial read

list_directory
^^^^^^^^^^^^^^
Browse repository directory structure.

**Parameters:**

- ``directory``: Directory path (default: .)
- ``recursive``: Show full tree (default: false)
- ``pattern``: File pattern filter (e.g., \*.py)

analyze_codebase
^^^^^^^^^^^^^^^^
Analyze the SYMFLUENCE codebase structure.

**Parameters:**

- ``depth``: Analysis depth - quick, detailed, or deep

propose_code_change
^^^^^^^^^^^^^^^^^^^
Propose a code modification (validates syntax, shows diff).

Uses fuzzy matching to find code even with minor whitespace differences.
Validates Python syntax before staging and shows a preview diff.

**Parameters:**

- ``file_path`` (required): Path to file to modify
- ``old_code`` (required): Exact code to replace (fuzzy matched)
- ``new_code`` (required): Replacement code
- ``description`` (required): Why this change is needed
- ``reason``: Type - bugfix, improvement, or feature
- ``fuzzy_threshold``: Match similarity (0.0-1.0, default 0.85)

**Example:**

.. code-block:: text

    Propose changing the timeout from 30 to 60 seconds in config.py

show_staged_changes
^^^^^^^^^^^^^^^^^^^
Display all staged changes with statistics and file summary.

Shows diff statistics (lines added/removed), list of modified files,
and the full diff. Provides guidance on next steps (run tests, commit, etc.).

**Parameters:** None

**Output includes:**

- Summary statistics (files changed, insertions, deletions)
- List of modified files
- Full diff output
- Next step guidance

run_tests
^^^^^^^^^
Run tests using pytest.

**Parameters:**

- ``test_pattern``: pytest pattern
- ``files``: Specific test files
- ``verbose``: Verbose output (default: false)

create_pr_proposal
^^^^^^^^^^^^^^^^^^
Create a PR proposal from staged changes.

Auto-detects modified files and generates a comprehensive PR description
with testing checklist. Provides step-by-step instructions for completing
the PR manually or via ``gh`` CLI.

**Parameters:**

- ``title`` (required): PR title
- ``description`` (required): PR body/description
- ``reason``: Type - bugfix, improvement, or feature

**Generated PR includes:**

- Summary section with your description
- Context-specific problem/feature/improvement section
- Auto-detected list of modified files
- Testing checklist
- Agent attribution footer

Meta Tools
----------

Tools for information and guidance.

show_help
^^^^^^^^^
Show help information about agent commands.

**Parameters:** None

list_available_tools
^^^^^^^^^^^^^^^^^^^^
List all available tools with descriptions.

**Parameters:** None

explain_workflow
^^^^^^^^^^^^^^^^
Explain the SYMFLUENCE workflow process and step sequence.

**Parameters:** None

Tool Categories Summary
-----------------------

======================  ======  ============================================
Category                Count   Purpose
======================  ======  ============================================
Workflow Steps          9       Execute individual workflow steps
Binary Management       4       Install and validate model binaries
Configuration           4       Manage configuration files
Workflow Management     4       Monitor and control workflows
Domain Setup            1       Create new modeling domains
Code Operations         7       Code analysis and modification
Meta Tools              3       Help and information
======================  ======  ============================================

See Also
--------

- :doc:`agent_guide` - Agent usage guide
- :doc:`configuration` - Configuration file reference
