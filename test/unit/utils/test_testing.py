import pytest
import os

from dolfin import mpi_comm_world, MPI

from muflon.utils.testing import GenericPostprocessorMMS

def test_GenericPostprocessorMMS():
    comm = mpi_comm_world()
    rank = MPI.rank(comm)

    p1 = GenericPostprocessorMMS()

    # Test that 'add_plot' raises for generic class
    with pytest.raises(NotImplementedError):
        p1.add_plot()

    # Create postprocessor
    p1 = GenericPostprocessorMMS()

    # Test add results
    results = [dict(a=1, b=2, c=3), dict(a=10, b=20, c=30)]
    for r in results:
        p1.add_result(rank, r)
    if rank > 0:
        assert not bool(p1.results)
    else:
        assert results == p1.results

    # Test popping
    for r in results:
        r.pop("c")
        assert len(r) == 2
    p1.pop_items(rank, ["c",])
    if rank > 0:
        assert not bool(p1.results)
    else:
        assert p1.results == results

    # Test writing the results into a binary files
    tempdir = "/tmp/pytest-of-fenics"
    fname = tempdir+"/foo.pickle"
    p1.flush_results(rank, fname)

    # Create another empty postprocessor
    p2 = GenericPostprocessorMMS()

    # Test reading the results from binary
    p2.read_results(rank, fname)
    if rank > 0:
        assert not bool(p2.results)
    else:
        assert p2.results == p1.results
