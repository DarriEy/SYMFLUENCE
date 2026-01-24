# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# Mock imports for autodoc - these packages have complex dependencies
# that are not needed for documentation generation
autodoc_mock_imports = [
    # Core scientific stack
    'numpy',
    'pandas',
    'scipy',
    'xarray',
    'pint_xarray',
    'distributed',
    'pyviscous',
    # Machine learning
    'torch',
    'sklearn',
    'scikit-learn',
    # Geospatial
    'gdal',
    'osgeo',
    'geopandas',
    'rasterio',
    'pyproj',
    'shapely',
    'fiona',
    'easymore',
    'rasterstats',
    'pvlib',
    'cdo',
    # Visualization
    'seaborn',
    'plotly',
    'matplotlib',
    'contextily',
    # Cloud and data access
    'gcsfs',
    'intake_xarray',
    'intake',
    'netCDF4',
    'h5netcdf',
    'cdsapi',
    's3fs',
    'cftime',
    # Hydrology specific
    'hydrobm',
    'baseflow',
    # Utilities
    'SALib',
    'psutil',
    'tqdm',
    'yaml',
    'requests',
    'numexpr',
    'bottleneck',
    'networkx',
    # AI/LLM
    'openai',
    # CLI
    'rich',
    # R integration
    'rpy2',
    # JAX (optional)
    'jax',
    'jaxlib',
    # Pydantic
    'pydantic',
    'pydantic_core',
    # Units
    'pint',
]

# Project information
project = 'SYMFLUENCE'
copyright = '2025, Darri Eythorsson'
author = 'Darri Eythorsson'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'myst_parser',  # For Markdown support
]

# Theme
html_theme = 'sphinx_rtd_theme'

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': False,
    'member-order': 'bysource',
}
# Skip private members by default
autodoc_default_flags = ['members', 'undoc-members']
# Better type hint formatting
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add support for both RST and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_context = {
    'display_github': True,
    'github_user': 'DarriEy',
    'github_repo': 'SYMFLUENCE',
    'github_version': 'main/docs/source/',
}
