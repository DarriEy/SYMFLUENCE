Getting Started
===============

This guide takes you from a fresh install to a complete simulation: the Bow
River at Banff, Alberta, modeled with SUMMA and ERA5 forcing.

Quick Start
-----------

.. code-block:: bash

   # Create a configuration from a built-in preset
   symfluence project init bow-river

   # Validate it
   symfluence config validate --config config_Bow_at_Banff.yaml

   # Run the full workflow
   symfluence workflow run --config config_Bow_at_Banff.yaml

``symfluence project list-presets`` lists the built-in presets. All data and
results land under the ``SYMFLUENCE_DATA_DIR`` set in your configuration
(default: ``~/symfluence_data``), in ``domain/<DOMAIN_NAME>/``.

The Workflow
------------

Every SYMFLUENCE run follows the same pipeline:

1. **Domain definition** — delineate the watershed and discretize it into HRUs
2. **Data acquisition** — download attributes (DEM, soil, land cover) and forcing data
3. **Preprocessing** — remap forcing to the domain and build model input files
4. **Simulation** — run the hydrological model, with routing where required
5. **Evaluation** — extract streamflow, compute metrics, generate plots

``symfluence workflow run`` executes the whole pipeline. Each stage can also be
run on its own with ``symfluence workflow step <step_name>``;
``symfluence workflow list-steps`` shows all steps and their aliases.

Walkthrough: Bow River at Banff
-------------------------------

The same run as the Quick Start, one stage at a time — useful for
understanding what each stage produces, and for restarting one that fails.

Prerequisites:

- SYMFLUENCE and its model binaries installed (see :doc:`installation`);
  verify with ``symfluence binary doctor``
- ~10 GB free disk space and an internet connection
- 1.5–3 hours end to end, most of it data download

**1. Create and check the configuration**

.. code-block:: bash

   symfluence project init bow-river

This writes ``config_Bow_at_Banff.yaml`` to the current directory. The preset
pins the essentials — domain, gauge, model, forcing, a one-year test period:

.. code-block:: yaml

   DOMAIN_NAME: "Bow_at_Banff"
   POUR_POINT_COORDS: "51.1722/-115.5717"
   HYDROLOGICAL_MODEL: "SUMMA"
   FORCING_DATASET: "ERA5"

Adjust anything you like, then validate:

.. code-block:: bash

   symfluence config validate --config config_Bow_at_Banff.yaml

**2. Set up the project and domain**

.. code-block:: bash

   symfluence workflow step setup_project --config config_Bow_at_Banff.yaml
   symfluence workflow step define_domain --config config_Bow_at_Banff.yaml
   symfluence workflow step discretize_domain --config config_Bow_at_Banff.yaml

``setup_project`` creates the directory tree under
``$SYMFLUENCE_DATA_DIR/domain/Bow_at_Banff/``. ``define_domain`` downloads a
DEM and delineates the catchment from the pour point (5–15 minutes);
``discretize_domain`` splits it into HRUs. Both write shapefiles to
``shapefiles/catchment/``.

**3. Acquire data**

.. code-block:: bash

   symfluence workflow step acquire_attributes --config config_Bow_at_Banff.yaml
   symfluence workflow step acquire_forcings --config config_Bow_at_Banff.yaml

Attributes (elevation, land cover, soil) go to ``attributes/``; forcing data
goes to ``forcing/raw_data/``. This is the long part — 30–90 minutes depending
on network and archive queues. ERA5 needs no credentials; CARRA/CERRA require
CDS API credentials (see :doc:`installation`).

**4. Preprocess**

.. code-block:: bash

   symfluence workflow step model_agnostic_preprocessing --config config_Bow_at_Banff.yaml
   symfluence workflow step model_specific_preprocessing --config config_Bow_at_Banff.yaml

The first step remaps forcing onto the catchment
(``forcing/basin_averaged_forcing/``); the second builds SUMMA's input and
settings files (``forcing/SUMMA_input/``, ``settings/SUMMA/``).

**5. Run and evaluate**

.. code-block:: bash

   symfluence workflow step run_model --config config_Bow_at_Banff.yaml
   symfluence workflow step postprocess_results --config config_Bow_at_Banff.yaml

``run_model`` executes SUMMA and routes the output with mizuRoute
(10–30 minutes for this domain). ``postprocess_results`` extracts streamflow to
``results/``, plots it, and prints baseline performance metrics (KGE, NSE,
bias) against the gauge observations. A low baseline score is normal — see
:doc:`calibration` for parameter optimization.

If a step fails, check the run log in
``domain/Bow_at_Banff/_workLog_Bow_at_Banff/`` and see :doc:`troubleshooting`.
``symfluence workflow status`` shows how far a run got, and
``symfluence workflow resume`` continues from the last completed step.

Next Steps
----------

- :doc:`calibration` — calibrate the model you just ran
- :doc:`configuration` — the full YAML reference; try other models
  (``HYDROLOGICAL_MODEL: FUSE``, ``GR``, ``HYPE``, ...)
- :doc:`examples` — progressive tutorials from point scale to continental
- :doc:`agent_guide` — drive all of the above with an AI agent
