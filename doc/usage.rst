User guide
==================

Introduction
---------------------------

The **ising** package allows to find a ground state (or, more generally, low energy spectrum) of an arbitrary spin-glass Ising model. That is, it allows one to find the minimum of the following energy function

.. math::

   H(s_0, \ldots, s_n) = - \sum_{i, j=0}^n J_{ij} s_i s_j - \sum_{i=0}^n h_i s_i

where :math:`J_{ij}` and :math:`h_i` are arbitrary real coefficients and variables :math:`s_i` under optimization are either :math:`-1` or :math:`1`.

Basic usage
-----------

The main functionallity of **ising** package is wrapped in ``ising.search`` function. As an example, suppose one would like to to find four lowest energy states given the following problem Hamiltonian


.. math::

   H(s_0, s_1, s_2) = -2s_0s_1 + 3s_1s_2 + 2.5s_2s_3 -s_0

In that case one can run ``ising.search`` as follows

.. code:: python

	  import ising

	  graph = {(0, 1): 2, (1, 2): -3, (2, 3): 2.5, (0, 0): 1}

	  result = ising.search(graph, num_states=4)
	  print(result.energies)

Note how the above model is specified as a dictionary:

- :math:`J_{ij}` are specified as entries with keys ``(i, j)``.
- :math:`h_i` are specified as (diagonal) entries with keys ``(i, i)``.

Other supported input formats
-----------------------------

There are three formats supported by **ising**:

- The dictionary format already presented in previous section.
- The *coefficients list format*. In this format coefficients are specified as a list of lists, in hich each row is of the form ``[i, j, J_ij]`` or ``[i, i, h_i]`.`
- The *matrix* format. In this format you specify your coefficients as a matrix in which diagonal elements correspond to :math:`h_i` and off-diagonal elements correspond to :math:`J_{ij}`. The matrix can either be a list of lists or a `numpy` array.

Putting it in another way, here are equivalent ways of specifying graph from the above basic example

.. code:: python

	  # coefficients list format
          graph = [[0, 1, 2], [1, 2, -3], [0, 0, 1], [2, 3, 2.5]],
	  # matrix format: as list of lists or numpy array
          graph = [[1, 2, 0, 0], [0, 0, -3, 0], [0, 0, 0, 2.5], [0, 0, 0, 0]],
          graph = np.array([[1, 2, 0, 0], [0, 0, -3, 0], [0, 0, 0, 2.5], [0, 0, 0, 0]]),

Note that the *matrix* format requires your spins to be labelled with :math:`0, \ldots, n`, other two formats are not restricted in this way.

Also note that since both :math:`J_{ij}` and :math:`J_{ji}` can be specified in all the formats it does not matter which one is chosen. In fact, if one chooses to specify both coefficients, both of them will be used. Therefore, using the following graphs would yield the same result as the previous example:

.. code:: python
	  
	  # coefficient list format
	  graph = [[0, 1, 1], [1, 0, 1], [1, 2, -3], [0, 0, 1], [2, 3, 2.5]],
	  # matrix format
	  graph = [[1, 1, 0, 0], [1, 0, -3, 0], [0, 0, 0, 2.5], [0, 0, 0, 0]]

Tweaking execution
------------------

You can use the following keyword arguments to ``ising.search`` to tweak its execution:

- ``num_states``: integer specifying how many low-energy states should be found.
- ``method``: indicating whether CPU (``method='CPU'``) or GPU (``method='GPU'``) implementation should be used. If not given, CPU implementation is used by default.
- ``energies_only``: boolean indicating whether to return only energies (``True``) or also states corresponding to those energies (``False``). Default is ``False``, set it to ``True`` if you don't need states, as it should shorten the execution time.
- ``chunk_size``: **ising** performs search in chunks of the size :math:`2^k`, where :math:`k` is choosen as a largest number such that computations are feasible on the host. You can tweak this value to use other exponent if you choose so.

In addition, for CPU implementation, one can specify how many OMP threads will be used for computations using ``OMP_NUM_THREADS`` environmental variable.
