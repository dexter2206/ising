Ising
============
\K. Jałowiecki, M. Rams and B. Gardas

Documentation: https://ising.readthedocs.io/en/latest/

**Ising** is an open source package for exactly solving abritrary Ising model instances via exhaustive search. It can be used as an excellent tool for benchmarking other solvers or generating low energy spectra. The package is compatible with \*NIX systems (and in principle should work on Windows too). **Ising** supports parallel computation via OpenMP or GPU, if it was build with CUDA support.

Build status
------------
|Build Status| |Documentation Status|


.. |Build Status| image:: https://travis-ci.org/dexter2206/ising.svg?branch=master
    :target: https://travis-ci.org/dexter2206/ising
.. |Documentation Status| image:: https://readthedocs.org/projects/ising/badge/?version=latest
    :target: https://ising.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Installation
-------------
If you are running Linux and are interested in CPU-only implementation, you can install **Ising** from Python Package Index.

.. code-block:: shell-session

   pip install ising

For other installation options, including building with CUDA support, please visit the official documentation_.

.. _documentation: https://ising.readthedocs.io/en/latest/
