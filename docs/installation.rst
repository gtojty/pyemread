Installation
============

pyemread requires Python 3.9 or later. Install it from GitHub::

    pip install git+https://github.com/gtojty/pyemread.git

The runtime dependencies (NumPy, pandas, Pillow and matplotlib) are declared
in ``pyproject.toml`` and listed in ``requirements.txt``. pyemread is tested
with pandas 1.5, 2.x and 3.x on Linux, macOS and Windows (see the continuous
integration workflow in ``.github/workflows/tests.yml``).

Rendering animations of eye movements (:func:`pyemread.gen.animate`) also
requires the FFmpeg program on the system path.

The example data (``examples/``) are not installed with the package; clone
the repository to use them::

    git clone https://github.com/gtojty/pyemread.git
    cd pyemread
    pip install -e ".[test]"
    pytest
