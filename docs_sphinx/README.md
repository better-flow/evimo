# Sphinx based documentation for EVIMO

To build:
```bash
pip3 install sphinx sphinx-rtd-theme myst-parser
make html
```

To autobuild and autoreload while editing:
```bash
pip3 install sphinx-autobuild
sphinx-autobuild docs_sphinx/source docs_sphinx/build/html
```

Copy build to the website directory
cp -r docs_sphinx/build/html docs/docs

The output directory is symlinked into the better-flow website as `evimo/docs/docs/`.
