import pytest
import os

from dolfin import mpi_comm_world, MPI

from muflon.utils.testing import GenericBenchPostprocessor

def test_GenericBenchPostprocessor():
    comm = mpi_comm_world()
    rank = MPI.rank(comm)

    # Create postprocessor
    p1 = GenericBenchPostprocessor()
    p1.register_fixed_variables((("a", 1), ("b", 2)))

    # Test that 'create_plots' raises for generic class
    if rank == 0:
        with pytest.raises(NotImplementedError):
            p1.create_plots(rank)
    else:
        p1.create_plots(rank)

    # Test add results
    results = [dict(a=1, b=2, c=3), dict(a=10, b=20, c=30)]
    for r in results:
        p1.add_result(rank, r)
    if rank > 0:
        assert not p1.results
    else:
        assert results == p1.results

    # Test popping
    for r in results:
        r.pop("c")
        assert len(r) == 2
    p1.pop_items(["c",])
    if rank > 0:
        assert not p1.results
    else:
        assert p1.results == results

    # Test writing the results into a binary files
    tempdir = "/tmp/pytest-of-fenics"
    fname = tempdir+"/foo.pickle"
    p1.save_results(fname)

    # Create another empty postprocessor
    p2 = GenericBenchPostprocessor()

    # Test reading the results from binary
    p2.read_results(rank, fname)
    if rank > 0:
        assert not p2.results
    else:
        assert p2.results == p1.results

    # Remove results before exiting (otherwise saved at destructor)
    p1.results = []
    p2.results = []
