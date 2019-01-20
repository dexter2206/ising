Installation
============

Installing binary wheel from PyPI
----------------------------------

If you are running a Linux system and are only interested in non GPU-enabled build, you can install
binary wheel from PyPI as usual::
		
   pip install ising

Unfortunately, we cannot provide GPU-enabled binary wheel due to a manylinux_ PEP-513 policy, as it is impossible to build **ising** on CentOS 5.

.. _manylinux: https://www.python.org/dev/peps/pep-0513/

Building from source
---------------------

If you are not running Linux and/or are interested in a GPU-enabled build, you need to build **ising** from source. The process is simple and requires running a single command. We highly recommend using virtual environment instead of installing the package into the global scope. Note that otherwise installing the package may require root privileges.

Prerequisites
+++++++++++++

To build **ising** you need the following:

- Virtually any C and C++ compiler,
- A Fortran compiler. The build script supports PGI, Intel, and gfortran compilers.
- The thrust_ library. This is installed by default alongside with CUDA toolkit, otherwise you need to make sure that thrust's include files are visible by your C++ compiler.
- ``numpy`` Python package installed in the same environment as is used to run the build process.

.. _thrust: https://thrust.github.io/
In addition, to build a GPU-enabled version you need PGI CUDA Fortran and compatible CUDA toolkit. Our package was tested against CUDA 9.2 and CUDA 10.0.

Building and installing
+++++++++++++++++++++++

To build the **ising** package download its source code and run ``install.py`` script as follows::

  python install.py --fcompiler=<fortran_compiler> [--usecuda]

where ``<fortran_compiler>`` is one of ``pgi``, ``intel``, ``gfortran``. The ``--usecuda`` switch can be used to enable GPU support. Note that ``--usecuda`` requires ``--fcompiler=pgi``.

The script should take care of building extensions and installing package, so after running the above command **ising** package should be ready to use. 


