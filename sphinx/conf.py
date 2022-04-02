# Configuration file for the Sphinx documentation builder

import os
import sys
import inspect
import importlib

sys.path.insert(0, os.path.abspath('..'))

## Project

project = 'PIQA'
copyright = '2020-2022, François Rozet'
repository = 'https://github.com/francois-rozet/piqa'

## Extensions

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
]

add_function_parentheses = False
autodoc_inherit_docstrings = False
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'
autosummary_generate = True

def linkcode_resolve(domain: str, info: dict, package: str = 'piqa') -> str:
    module = info.get('module', '')
    fullname = info.get('fullname', '')

    if not module or not fullname:
        return None

    objct = importlib.import_module(module)
    for name in fullname.split('.'):
        objct = getattr(objct, name)

    try:
        file = inspect.getsourcefile(objct)
        file = file[file.rindex(package):]

        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1
    except Exception as e:
        return None
    else:
        return f'{repository}/tree/docs/{file}#L{start}-L{end}'

napoleon_custom_sections = [
    'Original',
    'Wikipedia',
    ('Shapes', 'params_style'),
]

## HTML

default_role = 'python'
exclude_patterns = ['templates']
html_copy_source = False
html_css_files = [
    'custom.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
]
html_favicon = 'static/logo_dark.svg'
html_show_sourcelink = False
html_sourcelink_suffix = ''
html_static_path = ['static']
html_theme = 'furo'
html_theme_options = {
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/francois-rozet/piqa',
            'html': '<i class="fa-brands fa-github fa-lg"></i>',
            'class': '',
        },
    ],
    'light_css_variables': {
        'color-api-keyword': '#006699',
        'color-api-name': '#00aa88',
        'color-api-pre-name': '#00aa88',
    },
    'light_logo': 'logo.svg',
    'dark_css_variables': {
        'color-api-keyword': '#66d9ef',
        'color-api-name': '#a6e22e',
        'color-api-pre-name': '#a6e22e',
    },
    'dark_logo': 'logo_dark.svg',
    'sidebar_hide_name': True,
}
pygments_style = 'manni'
pygments_dark_style = 'monokai'
root_doc = 'piqa'
rst_prolog = """
.. role:: python(code)
    :class: highlight
    :language: python
"""
templates_path = ['templates']
