# Configuration file for the Sphinx documentation builder

import glob
import inspect
import importlib
import re
import subprocess
import piqa

## Project

package = 'piqa'
project = 'PIQA'
version = piqa.__version__
copyright = '2020-2023, François Rozet'
repository = 'https://github.com/francois-rozet/piqa'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

## Extensions

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
}
autodoc_inherit_docstrings = False
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'

autosummary_ignore_module_all = False

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable', None),
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}


def linkcode_resolve(domain: str, info: dict) -> str:
    module = info.get('module', '')
    fullname = info.get('fullname', '')

    if not module or not fullname:
        return None

    objct = importlib.import_module(module)
    for name in fullname.split('.'):
        objct = getattr(objct, name)

    try:
        file = inspect.getsourcefile(objct)
        file = file[file.rindex(package) :]

        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1
    except Exception as e:
        return None
    else:
        return f'{repository}/blob/{commit}/{file}#L{start}-L{end}'


napoleon_custom_sections = [
    'Original',
    'Wikipedia',
]

## Settings

add_function_parentheses = False
default_role = 'literal'
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
            'url': repository,
            'html': '<i class="fa-brands fa-github fa-lg"></i>',
            'class': '',
        },
    ],
    'light_css_variables': {
        'color-api-keyword': '#007020',
        'color-api-name': '#0e84b5',
        'color-api-pre-name': '#0e84b5',
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
html_title = f'{project} {version}'
pygments_style = 'sphinx'
pygments_dark_style = 'monokai'
rst_prolog = """
.. role:: py(code)
    :class: highlight
    :language: python
"""
templates_path = ['templates']

## Edit HTML

def edit_html(app, exception):
    if exception:
        raise exception

    for file in glob.glob(f'{app.outdir}/**/*.html', recursive=True):
        with open(file, 'r') as f:
            text = f.read()

        text = text.replace('<a class="muted-link" href="https://pradyunsg.me">@pradyunsg</a>\'s', '')
        text = text.replace('<span class="pre">[source]</span>', '<i class="fa-solid fa-code"></i>')
        text = re.sub(r'(<a class="reference external".*</a>)(<a class="headerlink".*</a>)', r'\2\1', text)

        with open(file, 'w') as f:
            f.write(text)

def setup(app):
    app.connect('build-finished', edit_html)
