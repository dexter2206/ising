Ising
============
\K. Jałowiecki, M. Rams and B. Gardas

**Ising** is an open source package for exactly solving abritrary Ising model instances via exhaustive search. It can be used as an excellent tool for benchmarking other solvers or generating low energy spectra. The package is compatible with \*NIX systems (and in principle should work on Windows too). **Ising** supports parallel computation via OpenMP or GPU, if it was build with CUDA support.

Build status
------------
.. image:: https://travis-ci.org/dexter2206/ising.svg?branch=master
    :target: https://travis-ci.org/dexter2206/ising

Installation
-------------
If you are running Linux and are interested in CPU-only implementation, you can install **Ising** from Python Package Index.

```text
pip install ising
```

For other installation options, including building with CUDA support, please visit the official documentation.
