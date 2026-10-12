# Authors and contributors

**Tao Gong** develops and maintains pyemread. The code was written with
substantial assistance from a generative AI tool, under her direction and
review, as declared in the accompanying article. Apart from the adapted
`slice` implementation (below), no other person wrote code for the package.

**David Braze**, the author's supervisor at Haskins Laboratories, wrote the
in-house C/C++ programs for eye-movement analysis from which pyemread was
originally derived, including the definitions of the word-level reading
measures. He did not write code for this package.

**Jonathan Gordils** assisted with the collection of the example data, and
**Hosung Nam**, a visiting scholar at Haskins Laboratories, commented on the
calculation of the reading measures. Neither wrote code for this package.

Jonathan Gordils and Hosung Nam have agreed to be acknowledged.

The `slice` line-assignment method in `pyemread/lines.py` is adapted from
the Python implementation by Jon W. Carr and colleagues
(https://github.com/jwcarr/drift, CC BY 4.0) of the algorithm of Glandorf and
Schroeder (2021).
