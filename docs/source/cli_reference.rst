.. _cli_reference:

=========================================
CLI Reference
=========================================

SYMFLUENCE provides a comprehensive command-line interface for managing hydrological
modeling workflows. This reference documents all available commands, options, and usage patterns.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The CLI follows a two-level hierarchical structure:

- **Level 1**: Command categories (workflow, project, binary, config, job, example, agent)
- **Level 2**: Specific actions within each category

Basic usage:

.. code-block:: bash

   symfluence <category> <command> [options]

Global Options
==============

These options are available to all commands:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Description
   * - ``--config PATH``
     - Path to configuration file (default: ./config.yaml)
   * - ``--debug``
     - Enable debug output and stack traces
   * - ``--visualise / --visualize``
     - Enable visualization during execution
   * - ``--dry-run``
     - Show what would be executed without running
   * - ``--profile``
     - Enable I/O profiling
   * - ``--profile-output PATH``
     - Path for profiling report (default: profile_report.json)
   * - ``--profile-stacks``
     - Capture stack traces in profiling
   * - ``--version``
     - Display SYMFLUENCE version

Workflow Commands
=================

Manage and execute SYMFLUENCE workflows.

workflow run
------------

Execute the complete workflow from start to finish.

.. code-block:: bash

   symfluence workflow run [--config CONFIG] [--force-rerun] [--continue-on-error]

**Options:**

- ``--force-rerun``: Force rerun of all steps (skip caching)
- ``--continue-on-error``: Continue executing steps on errors

**Example:**

.. code-block:: bash

   symfluence workflow run --config my_config.yaml --force-rerun

workflow step
-------------

Execute a single workflow step.

.. code-block:: bash

   symfluence workflow step STEP_NAME [--config CONFIG] [--force-rerun]

**Available Steps:**

1. ``setup_project`` - Initialize project structure
2. ``create_pour_point`` - Create pour point shapefile
3. ``acquire_attributes`` - Download geospatial attributes
4. ``define_domain`` - Define hydrological domain
5. ``discretize_domain`` - Discretize into HRUs
6. ``process_observed_data`` - Process observations
7. ``acquire_forcings`` - Acquire meteorological forcing
8. ``model_agnostic_preprocessing`` - Model-agnostic preprocessing
9. ``model_specific_preprocessing`` - Model-specific setup
10. ``run_model`` - Execute hydrological model
11. ``calibrate_model`` - Run parameter calibration
12. ``run_emulation`` - Run emulation optimization
13. ``run_benchmarking`` - Run benchmarking analysis
14. ``run_decision_analysis`` - Run decision analysis
15. ``run_sensitivity_analysis`` - Run sensitivity analysis
16. ``postprocess_results`` - Postprocess results

**Example:**

.. code-block:: bash

   symfluence workflow step calibrate_model --config config.yaml

workflow steps
--------------

Execute multiple workflow steps in sequence.

.. code-block:: bash

   symfluence workflow steps STEP1 STEP2 ... [--config CONFIG] [--force-rerun]

**Example:**

.. code-block:: bash

   symfluence workflow steps acquire_forcings model_agnostic_preprocessing run_model

workflow status
---------------

Show workflow execution status.

.. code-block:: bash

   symfluence workflow status [--config CONFIG]

workflow validate
-----------------

Validate configuration file syntax.

.. code-block:: bash

   symfluence workflow validate [--config CONFIG]

workflow list-steps
-------------------

List all available workflow steps.

.. code-block:: bash

   symfluence workflow list-steps

workflow resume
---------------

Resume workflow from a specific step.

.. code-block:: bash

   symfluence workflow resume STEP_NAME [--config CONFIG] [--force-rerun]

**Example:**

.. code-block:: bash

   symfluence workflow resume calibrate_model --config config.yaml

workflow clean
--------------

Clean intermediate or output files.

.. code-block:: bash

   symfluence workflow clean [--config CONFIG] [--level LEVEL] [--dry-run]

**Options:**

- ``--level``: Cleaning level (intermediate, outputs, all; default: intermediate)
- ``--dry-run``: Preview what would be cleaned

**Example:**

.. code-block:: bash

   symfluence workflow clean --level all --dry-run

Project Commands
================

Initialize projects and configure pour points.

project init
------------

Initialize a new SYMFLUENCE project.

.. code-block:: bash

   symfluence project init [PRESET] [options]

**Options:**

- ``--domain TEXT``: Domain name
- ``--model {SUMMA,FUSE,GR,HYPE,MESH,RHESSys,NGEN,LSTM}``: Model selection
- ``--start-date YYYY-MM-DD``: Simulation start date
- ``--end-date YYYY-MM-DD``: Simulation end date
- ``--forcing TEXT``: Forcing dataset
- ``--discretization TEXT``: Discretization method
- ``--definition-method TEXT``: Domain definition method
- ``--output-dir PATH``: Output directory (default: ./)
- ``--scaffold``: Create full directory structure
- ``--minimal``: Create minimal configuration (10 required fields)
- ``--comprehensive``: Create comprehensive configuration (400+ options)
- ``-i, --interactive``: Run interactive configuration wizard

**Examples:**

.. code-block:: bash

   # Interactive setup with scaffold
   symfluence project init --interactive --scaffold

   # With preset
   symfluence project init fuse-provo --model FUSE --start-date 2020-01-01

   # Minimal config
   symfluence project init --domain MyDomain --minimal

project pour-point
------------------

Set up pour point workflow.

.. code-block:: bash

   symfluence project pour-point LAT/LON --domain-name NAME --definition METHOD [options]

**Arguments:**

- ``coordinates``: Pour point as "lat/lon" (e.g., 51.1722/-115.5717)

**Required Options:**

- ``--domain-name NAME``: Domain/watershed name
- ``--definition METHOD``: Definition method (lumped, point, subset, delineate)

**Optional:**

- ``--bounding-box``: Custom bounding box (LAT_MAX/LON_MIN/LAT_MIN/LON_MAX)
- ``--experiment-id``: Override experiment ID
- ``--output-dir``: Output directory

**Example:**

.. code-block:: bash

   symfluence project pour-point 51.1722/-115.5717 --domain-name Bow --definition delineate

project list-presets
--------------------

List available initialization presets.

.. code-block:: bash

   symfluence project list-presets

project show-preset
-------------------

Show details of a specific preset.

.. code-block:: bash

   symfluence project show-preset PRESET_NAME

Binary Commands
===============

Install and manage external tools.

binary install
--------------

Install external tools.

.. code-block:: bash

   symfluence binary install [TOOL1 TOOL2 ...] [--force]

**Available Tools:**

- ``summa`` - SUMMA hydrological model
- ``mizuroute`` - mizuRoute routing model
- ``fuse`` - FUSE hydrological model
- ``hype`` - HYPE hydrological model
- ``mesh`` - MESH model
- ``taudem`` - TauDEM terrain analysis
- ``gistool`` - GIS analysis tool
- ``datatool`` - Data processing tool
- ``rhessys`` - RHESSys ecosystem model
- ``ngen`` - NextGen framework
- ``ngiab`` - NextGen in a Box
- ``sundials`` - SUNDIALS ODE solver

**Example:**

.. code-block:: bash

   symfluence binary install summa mizuroute taudem --force

binary validate
---------------

Validate installed binaries.

.. code-block:: bash

   symfluence binary validate [--verbose]

binary doctor
-------------

Run comprehensive system diagnostics.

.. code-block:: bash

   symfluence binary doctor

binary info
-----------

Display information about installed tools.

.. code-block:: bash

   symfluence binary info

Config Commands
===============

Manage and validate configuration files.

config list-templates
---------------------

List available configuration templates.

.. code-block:: bash

   symfluence config list-templates

config validate
---------------

Validate configuration file.

.. code-block:: bash

   symfluence config validate [--config CONFIG]

config validate-env
-------------------

Validate system environment.

.. code-block:: bash

   symfluence config validate-env

Job Commands
============

Submit workflows to HPC clusters via SLURM.

job submit
----------

Submit workflow as SLURM job.

.. code-block:: bash

   symfluence job submit [options] [WORKFLOW_COMMAND ...]

**Options:**

- ``--name NAME``: SLURM job name
- ``--time TIME``: Time limit (HH:MM:SS; default: 48:00:00)
- ``--nodes N``: Number of nodes (default: 1)
- ``--tasks N``: Number of tasks (default: 1)
- ``--memory MEM``: Memory requirement (default: 50G)
- ``--account ACCOUNT``: Account to charge
- ``--partition PARTITION``: Partition/queue name
- ``--modules MODULES``: Module to restore (default: symfluence_modules)
- ``--conda-env ENV``: Conda environment (default: symfluence)
- ``--wait``: Monitor job until completion
- ``--template PATH``: Custom SLURM template

**Examples:**

.. code-block:: bash

   # Simple submission
   symfluence job submit workflow run --config config.yaml

   # With resources
   symfluence job submit --time 72:00:00 --nodes 4 --tasks 16 \
     --account myaccount workflow run --config config.yaml

   # With monitoring
   symfluence job submit --wait workflow run --config config.yaml

Example Commands
================

Launch and manage example notebooks.

example launch
--------------

Launch an example notebook.

.. code-block:: bash

   symfluence example launch EXAMPLE_ID [--lab] [--notebook]

**Options:**

- ``--lab``: Launch in JupyterLab (default)
- ``--notebook``: Launch in classic Jupyter Notebook

example list
------------

List available example notebooks.

.. code-block:: bash

   symfluence example list

Agent Commands
==============

Interactive AI agent interface.

agent start
-----------

Start interactive AI agent mode.

.. code-block:: bash

   symfluence agent start [--config CONFIG] [--verbose]

agent run
---------

Execute a single agent prompt.

.. code-block:: bash

   symfluence agent run PROMPT [--config CONFIG] [--verbose]

**Example:**

.. code-block:: bash

   symfluence agent run "What is the current SUMMA configuration?" --config config.yaml

Exit Codes
==========

All commands return standardized exit codes:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Code
     - Name
     - Meaning
   * - 0
     - SUCCESS
     - Command completed successfully
   * - 1
     - GENERAL_ERROR
     - General error
   * - 2
     - USAGE_ERROR
     - Invalid arguments/usage
   * - 3
     - CONFIG_ERROR
     - Configuration file issues
   * - 4
     - VALIDATION_ERROR
     - Input validation failed
   * - 5
     - FILE_NOT_FOUND
     - Required file missing
   * - 6
     - DIRECTORY_NOT_FOUND
     - Required directory missing
   * - 7
     - BINARY_ERROR
     - External binary not found
   * - 8
     - BINARY_BUILD_ERROR
     - Failed to build binary
   * - 9
     - NETWORK_ERROR
     - Network/download failure
   * - 10
     - PERMISSION_ERROR
     - Permission denied
   * - 11
     - TIMEOUT_ERROR
     - Operation timed out
   * - 12
     - DEPENDENCY_ERROR
     - Missing dependency
   * - 13
     - MODEL_ERROR
     - Model execution error
   * - 14
     - WORKFLOW_ERROR
     - Workflow execution error
   * - 15
     - DATA_ERROR
     - Data processing error
   * - 20
     - JOB_SUBMIT_ERROR
     - SLURM submission failed
   * - 21
     - JOB_EXECUTION_ERROR
     - Job execution failed
   * - 130
     - USER_INTERRUPT
     - User pressed Ctrl+C
   * - 143
     - SIGTERM
     - Process terminated

Common Usage Patterns
=====================

Quick Project Setup
-------------------

.. code-block:: bash

   # Interactive setup
   symfluence project init --interactive --scaffold

   # Or with preset
   symfluence project init fuse-provo --scaffold

Complete Workflow
-----------------

.. code-block:: bash

   # Validate first
   symfluence workflow validate --config config.yaml

   # Run everything
   symfluence workflow run --config config.yaml

Selective Steps
---------------

.. code-block:: bash

   # Single step
   symfluence workflow step calibrate_model --config config.yaml

   # Multiple steps
   symfluence workflow steps acquire_forcings run_model

   # Resume from step
   symfluence workflow resume calibrate_model --config config.yaml

HPC Submission
--------------

.. code-block:: bash

   # Save environment
   module save symfluence_modules

   # Submit with monitoring
   symfluence job submit --time 72:00:00 --nodes 4 --wait \
     workflow run --config config.yaml

Debugging
---------

.. code-block:: bash

   # Enable profiling
   symfluence workflow run --config config.yaml --profile --profile-stacks

   # Debug mode
   symfluence workflow run --config config.yaml --debug

Getting Help
============

.. code-block:: bash

   # General help
   symfluence --help

   # Category help
   symfluence workflow --help

   # Command help
   symfluence workflow run --help
