Ising
============
\K. Jałowiecki, M. Rams and B. Gardas

Documentation: https://ising.readthedocs.io/en/latest/

**Ising** is an open source package to solve small but otherwise abritrary spin-glass Ising models using exhaustive (brute force) search. It can serve as an excellent tool for benchmarking other solvers or generating low energy spectra (desirable e.g. for machine learning related tasks). The package is compatible with \*NIX systems (and in principle should work on Windows too). **Ising** supports parallel computation via OpenMP or GPU, provided it has been build with CUDA support.

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

Usage example
--------------
The below example finds 4 lowest energy states of the Ising model defined by

.. math::

   H(s_0, s_1, s_2) = -2s_0s_1 + 3s_1s_2 + 2.5s_2s_3 -s_0
   
.. code:: python

	  import ising

	  graph = {(0, 1): 2, (1, 2): -3, (2, 3): 2.5, (0, 0): 1}

	  result = ising.search(graph, num_states=4)
	  print(result.energies)
      
For advanced usage, including GPU support and tweaking execution parameters see documentation_.
