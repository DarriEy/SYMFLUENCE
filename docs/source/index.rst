.. image:: _static/Conf.jpg
   :width: 200px
   :align: center

SYMFLUENCE Documentation
========================

**Earth-system modeling. Simplified.**

---

Overview
--------

**SYMFLUENCE** is a computational framework for Earth-system modeling that integrates data preparation, model setup, calibration, and evaluation in unified workflows.

**Core capabilities:**
- Multi-model integration (SUMMA, FUSE, HBV, NextGen, GR4J, LSTM, mizuRoute)
- Automated parameter calibration and optimization
- Parallel execution and HPC support
- Reproducible workflows from data to results
- Comprehensive evaluation and reporting

---

Quick Start
-----------

- :doc:`installation` — Installation and environment setup
- :doc:`getting_started` — Your first project workflow
- :doc:`examples` — Tutorials and case studies
- `GitHub Repository <https://github.com/DarriEy/SYMFLUENCE>`_

---

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   getting_started
   configuration
   calibration
   examples

.. toctree::
   :maxdepth: 2
   :caption: AI Agent

   agent_guide
   agent_tools

.. toctree::
   :maxdepth: 2
   :caption: Hydrological Models

   models/model_summa
   models/model_fuse
   models/model_gr
   models/model_hbv
   models/model_hype
   models/model_ngen
   models/model_lstm
   models/model_gnn
   models/model_mesh
   models/model_rhessys

.. toctree::
   :maxdepth: 2
   :caption: Routing Models

   routing/mizuroute
   routing/troute

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   config_system
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   architecture
   developer_guide

.. toctree::
   :maxdepth: 1
   :caption: Project Information

   changelog

---

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
