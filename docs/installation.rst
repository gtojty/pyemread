Installation
============

pyemread requires Python 3.9 or later. Install it from GitHub::

    pip install git+https://github.com/gtojty/pyemread.git

The runtime dependencies (NumPy, pandas, Pillow and matplotlib) are declared
in ``pyproject.toml`` and listed in ``requirements.txt``. pyemread is tested
on Linux with Python 3.9 to 3.13 (latest compatible dependencies, and pandas 1.5
and 2.2) and on macOS and Windows with Python 3.12 (see the continuous
integration workflow in ``.github/workflows/tests.yml``).

Rendering animations of eye movements (:func:`pyemread.gen.animate`) also
requires the FFmpeg program on the system path.

The example data (``examples/``) are not installed with the package; clone
the repository to use them::

    git clone https://github.com/gtojty/pyemread.git
    cd pyemread
    pip install -e ".[test]"
    pytest
