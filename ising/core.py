"""Find states of lowest energy in Ising Model."""
import logging
import numpy
import progressbar
import psutil
from ising.helpers import EnergiesAndStates, decode_state, ising_to_qubo
from ising.graphtools import read_graph
from isingcpu import cpusearch

if not logging.getLogger('ising').handlers:
    logging.basicConfig(level='WARNING')

try:
    from isinggpu import gpusearch
except ImportError:
    logging.getLogger('ising').info('The ising package is installed with no GPU support.')
    gpusearch = None

def search(graph, num_states=10, energies_only=False, **kwargs):
    """Find lowest energies and corresponding states of Ising Model specified by Jh.

    :param Jh: matrix encoding hamiltonian of the sytem. Should be real and symmetric.
    :type Jh: numpy.ndarray or array-like
    :param sweep_size: number of bits being changed during a single sweep during exhaustive
     search. Has to obey `1 <= sweep_size <= Jh.shape[0]`.
    :type sweep_size: int
    :param num_states: how many lowest states to keep. Default is 10.
    :type num_states:
    :returns: namedtuple with components `states` and `energies` holding computed states of
     lowest energy and corresponding energies.
    :rtype: :py:class:`EnergiesAndStates
    """
    logger = logging.getLogger('ising')

    jh_mat, labels = read_graph(graph)

    qubo, const = ising_to_qubo(jh_mat)
    method = choose_method(gpusearch, **kwargs)
    logger.info('Choosen computation mehtod: %s', method)

    if 'chunk_size' in kwargs:
        chunk_size = kwargs['chunk_size']
        logger.debug('Chunk size given explicitly as %f.', chunk_size)
    else:
        logger.debug('Deducing chunk size from the available memory.')
        chunk_size = max_chunk_size_for_method(method)

    if chunk_size > jh_mat.shape[0]:
        logger.debug('Clipping chunk size to the number of spins.')
        chunk_size = jh_mat.shape[0]

    logger.info('Chunk size exponent that will be used: %d', chunk_size)

    if num_states > 2 * 2 ** chunk_size:
        logger.warning('Requested more states than two chunks. Clipping at 2 ** (chunk size)')
        num_states = 2 ** chunk_size

    if kwargs.get('show_progress', False):
        pbar = progressbar.ProgressBar(max_value=2 ** (jh_mat.shape[0]-chunk_size))
        callback = pbar.update
    else:
        pbar = None
        callback = dummy_callback

    if method.lower() == 'cpu':
        logger.info('Running CPU search: Jh.shape=%s, chunk_size=%d, num_states=%d',
                    jh_mat.shape, chunk_size, num_states)
        callback(0)
        if energies_only:
            energies = cpusearch.find_lowest_energies_only(qubo,
                                                           chunk_size,
                                                           num_states,
                                                           callback=callback)
        else:
            energies, state_reprs = cpusearch.find_lowest(qubo,
                                                          chunk_size,
                                                          num_states,
                                                          callback=callback)
    else:
        logger.info('Running GPU search: Jh.shape=%s, num_states=%d, chunk_size=%d',
                    jh_mat.shape, num_states, chunk_size)
        callback(0)
        if energies_only:
            energies = gpusearch.find_lowest_energies_only(qubo,
                                                           chunk_size,
                                                           num_states,
                                                           callback=callback)
        else:
            energies, state_reprs = gpusearch.find_lowest(qubo,
                                                          chunk_size,
                                                          num_states,
                                                          callback=callback)
    if pbar is not None:
        pbar.finish()
    if energies_only:
        state_reprs = None
    return EnergiesAndStates(states=state_reprs, energies=energies + const)

def dummy_callback(_):
    """Dummy callback doing nothing."""
    pass

def choose_method(gpusearch, **kwargs):
    """Choose computation method judging by user supplied args and availability of extensions.

    :param gpusearch: the imported gpusearch Fortran extension, or None if no such extension
     was available.
    :type gpusearch: Fortran module or None
    :param kwargs: keyword arguments passed by the user to the :py:func:`search` function.
    :type kwargs: dict
    :returns: String with computation method choosen (either 'cpu' or 'gpu').
    :rtype: str
    :raises ValueError: if the 'method' kwarg was supplied but is not equal to 'cpu' or 'gpu'
    (case insensitive) OR 'gpu' computation method was requested but no GPU extension for
    the `ising` package has been built and instaled.

    .. notes::
       The only kwarg supported is the "method" kwarg, which can override automatic selection
       of computation method.
    """
    logger = logging.getLogger('ising')
    supplied_method = kwargs.get('method')
    if supplied_method is None:
        logger.info('No computation method supplied. Determining computation method myself.')
        return 'gpu' if gpusearch is not None else 'cpu'
    supplied_method = supplied_method.lower()
    if supplied_method not in ('cpu', 'gpu'):
        raise ValueError('Unknown computation method supplied: %s', supplied_method)
    if supplied_method == 'gpu' and gpusearch is None:
        raise ValueError('Computation via GPU implementation requested but extension with GPU '
                         'implementation was not build and installed. If your setup is equipped '
                         'with CUDA supporting GPU rebuild with --usecuda flag and reinstall '
                         'the package.')
    return supplied_method

def determine_max_sweep_size():
    available = psutil.virtual_memory().available

    # Fortran code should be able to allocate 2 ** sweep_size array of integers
    # and floating point numbers (8 bytes for each element in each array)
    # We also need to take into account for some margin of output states and
    # auxiliary variables

    # This many numbers can be stored in the memory.
    elements_max = available // 16

    # Determine the highest bit set - that is the maximum sweep size (w/o accounting
    # for a margin.
    sweep = 0
    while elements_max != 0:
        elements_max >>= 1
        sweep += 1

    # Fixed margin of 500 MB
    if available - 2 ** sweep * 8 * 2 < 500 * 1024 * 1024:
        sweep -= 1

    return sweep

def max_chunk_size(mem_bytes):
    """Determine maximum chunk size for use with given ammount of memory.

    :param mem_bytes: size of available memory in bytes. This might mean different things
     depending on for what computation method the chunk_size is being computed but the
     bottom line is: this is the ammount of memory that we are able to use
    :type mem_bytes: int
    :returns: maximum chunk size as a exponent determining part of search spaced sweep
     during a single iteration. More precisely, if the returned value is `m` then
     2 ** `m` states can be searched during a single pass of the algorithm (and this is
     a maximal such exponent `m` that can be used).
    :rtype: int
    """
    elements_max = mem_bytes // 16 // 2
    chunk_size = 0
    while elements_max > 1:
        elements_max >>= 1
        chunk_size += 1
    if 2 * 16 * 2 ** chunk_size + 1024 * 1024 * 1024 > mem_bytes:
        chunk_size -= 1
    return chunk_size

def max_chunk_size_for_method(method):
    """Determine max chunk size given computation method.

    :param method: method that shall be used for computations, either 'cpu' or 'gpu' (case
     insensitive).
    :type method: str
    :returns: maximum chunk size for given computation method
    :rtype: int
    :raises ValueError: if unrecognized computation method is passed.
    """
    method = method.lower()
    if method == 'cpu':
        mem_bytes = psutil.virtual_memory().available
    elif method == 'gpu':
        mem_bytes = gpusearch.get_device_properties()
    else:
        raise ValueError('Unknown computation method: %s', method)
    return max_chunk_size(mem_bytes)
