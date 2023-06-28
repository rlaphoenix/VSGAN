# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from datetime import datetime

from dunamai import Style, Version

# -- Project information -----------------------------------------------------

project = 'VSGAN'
copyright = f'2019-{datetime.now().year}, rlaphoenix'
author = 'rlaphoenix'

version = Version.from_git().serialize(style=Style.SemVer)
release = Version.from_git().base

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'm2r2',
    'sphinxcontrib.youtube',
    'sphinxcontrib.images',
]

master_doc = 'index'

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]

templates_path = ['_templates']

autodoc_mock_imports = ['vapoursynth']

# -- Options for internationalization ----------------------------------------

language = 'en'

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True
}
html_logo = '_static/images/icon.png'
html_static_path = ['_static']

html_css_files = [
    'styles/custom.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/fontawesome.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/solid.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/brands.min.css',
]

html_js_files = [
    'js/sphinxcontrib-images.js'
]

html_sidebars = {
    '**': [
        'sidebar/scroll-start.html',
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/navigation.html',
        'sidebar/scroll-end.html',
    ]
}
