"""Find states of lowest energy in Ising Model."""
import logging
import numpy
import progressbar
import psutil
from ising.helpers import EnergiesAndStates, decode_state, ising_to_qubo
from ising.graphtools import read_graph
import isingcpu

if not logging.getLogger("ising").handlers:
    logging.basicConfig(level="WARNING")

try:
    import isinggpu
except ImportError:
    logging.getLogger("ising").info(
        "The ising package is installed with no GPU support."
    )
    isinggpu = None


def search(
    graph,
    num_states=10,
    energies_only=False,
    use_gpu=False,
    block_size=32,
    chunk_exponent=None,
    precision="single",
    show_progress=False,
):
    if use_gpu and isinggpu is None:
        raise ValueError(
            "The ising package has been installed without GPU support. Set use_gpu=False"
        )

    logger = logging.getLogger("ising")
    jh_mat, labels = read_graph(graph)
    qubo, const = ising_to_qubo(jh_mat)
    kwargs = {}

    method = "gpu" if use_gpu else "cpu"

    if chunk_exponent is not None:
        logger.debug("Chunk size given explicitly as %f.", chunk_exponent)
    else:
        logger.debug("Deducing chunk size from the available memory.")
        chunk_exponent = max_chunk_size_for_method(method)

    if chunk_exponent > jh_mat.shape[0]:
        logger.debug("Clipping chunk size exponent to the number of spins.")
        chunk_exponent = jh_mat.shape[0]

    if use_gpu:
        kwargs["block_size"] = block_size
        kwargs["grid_size"] = int(numpy.ceil(2 ** (chunk_exponent) / block_size))
        module = isinggpu
    else:
        module = isingcpu

    logger.info("Choosen computation mehtod: %s", method)

    logger.info("Chunk size exponent that will be used: %d", chunk_exponent)

    if num_states > 2 * 2 ** chunk_exponent:
        logger.warning(
            "Requested more states than two chunks. Clipping at 2 ** (chunk size)"
        )
        num_states = 2 ** chunk_exponent

    dtype = numpy.float32 if precision == "single" else numpy.float64

    kwargs["Q"] = numpy.array(qubo, dtype=dtype)
    kwargs["show_progress"] = show_progress
    kwargs["num_spins"] = qubo.shape[0]
    kwargs["chunk_exponent"] = chunk_exponent
    kwargs["energies_out"] = numpy.empty(num_states, dtype=dtype)
    kwargs["num_states"] = num_states

    if precision == "single" and energies_only:
        find = module.cy_find_lowest_energies_only_float
    elif precision == "single":
        find = module.cy_find_lowest_float
        kwargs["states_out"] = numpy.empty(num_states, dtype=numpy.int64)
    elif energies_only:
        find = module.cy_find_lowest_energies_only_double
    else:
        find = module.cy_find_lowest_double
        kwargs["states_out"] = numpy.empty(num_states, dtype=numpy.int64)

    find(**kwargs)
    states_out = None if energies_only else kwargs["states_out"]

    return EnergiesAndStates(
        raw_states=states_out, labels=labels, energies=kwargs["energies_out"] + const
    )


def dummy_callback(_):
    """Dummy callback doing nothing."""
    pass


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
    if method == "cpu":
        mem_bytes = psutil.virtual_memory().available
    elif method == "gpu":
        mem_bytes, _ = isinggpu.cy_get_gpu_mem_info()
    else:
        raise ValueError("Unknown computation method: %s", method)
    return max_chunk_size(mem_bytes)
