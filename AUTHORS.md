# Authors and contributors

**Tao Gong** wrote and maintains the Python code in this package, with the
assistance of a generative AI tool, as declared in the accompanying article.
No other person wrote code for the package; the one exception is the adapted
`slice` implementation described below.

**David Braze**, the author's supervisor at Haskins Laboratories, wrote the
in-house C/C++ programs for eye-movement analysis from which pyemread was
originally derived, including the definitions of the word-level reading
measures. He did not write code for this package.

Jonathan Gordils and Hosung Nam are credited in the module metadata of earlier
versions for their contributions to that earlier work.

The `slice` line-assignment method in `pyemread/lines.py` is adapted from
the Python implementation by Jon W. Carr and colleagues
(https://github.com/jwcarr/drift, CC BY 4.0) of the algorithm of Glandorf and
Schroeder (2021).
