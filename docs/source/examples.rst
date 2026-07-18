Examples
========

Ready-to-run examples — configuration, Jupyter notebook, and optional SLURM
script — live in the ``examples/`` directory of the repository. They progress
from simple to advanced:

1. **Point scale (01)** — snow and energy balance validation (SNOTEL, FLUXNET)
2. **Basin scale (02)** — Bow River case studies (lumped, semi-distributed, distributed)
3. **Regional and continental (03)** — Iceland and North America workflows
4. **Workshops (04)** — guided hands-on exercises (Logan River, Provo River)

Running the Examples
--------------------

The examples require a source checkout (they are not shipped with the pip
package):

.. code-block:: bash

   git clone https://github.com/symfluence-org/SYMFLUENCE.git
   cd SYMFLUENCE
   ./scripts/symfluence-bootstrap --install

From the repository root, list and launch notebooks by ID:

.. code-block:: bash

   symfluence example list
   symfluence example launch 02b

Or open one directly:

.. code-block:: bash

   cd examples/02_watershed_modelling/
   jupyter lab 02b_basin_semi_distributed.ipynb

Start with ``01a_point_scale_snotel.ipynb`` to learn configuration and
validation, then work upward. Each notebook explains its own workflow.

References
----------

- Example notebooks: `examples/ <https://github.com/symfluence-org/SYMFLUENCE/tree/main/examples>`_
- Configuration templates: `config_templates/ <https://github.com/symfluence-org/SYMFLUENCE/tree/main/src/symfluence/resources/config_templates>`_
- :doc:`configuration` — configuration reference
