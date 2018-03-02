import pytest
import gc
from dolfin import MPI, mpi_comm_world

def pytest_addoption(parser):
    parser.addoption("--case", action="store", type=int,
                     choices=[1, 2], default=1,
                     help="lower (1) vs. higher (2) density ratio")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.case
    if 'case' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("case", [option_value])

# Configuration from dolfin
def pytest_runtest_teardown(item):
    """Collect garbage after every test to force calling
    destructors which might be collective"""

    # Do the normal teardown
    item.teardown()

    # Collect the garbage (call destructors collectively)
    del item
    # NOTE: How are we sure that 'item' does not hold references
    #       to temporaries and someone else does not hold a reference
    #       to 'item'?! Well, it seems that it works...
    gc.collect()
    MPI.barrier(mpi_comm_world())
