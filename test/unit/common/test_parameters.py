from dolfin import mpi_comm_world, MPI, set_log_level, DEBUG, INFO
from muflon.common.parameters import mpset, _MuflonParameterSet

import os

def test_mpset():
    #set_log_level(DEBUG)

    # Print parameters and their values
    #mpset.show()

    # Try to add parameter
    mpset.add("foo", "bar")
    assert mpset["foo"] == "bar"

    # Try direct access to a parameter
    mpset["foo"] = "bar_"
    assert mpset["foo"] == "bar_"

    # Try to write parameters to a file
    comm = mpi_comm_world()
    tempdir = "/tmp/pytest-of-fenics"
    fname = tempdir+"/foo.xml"
    mpset.write(comm, fname)
    if MPI.rank(comm) == 0:
        assert os.path.isfile(fname)
    MPI.barrier(comm) # wait until the file is written

    # Change back value of parameter 'foo'
    mpset["foo"] = "bar"
    assert mpset["foo"] == "bar"

    # Try to read parameters back
    mpset.read(fname)
    assert mpset["foo"] == "bar_"
    MPI.barrier(comm) # wait until each process finishes reading
    if MPI.rank(comm) == 0:
        os.remove(fname)
    del fname

    # Check that every other call points to the same object
    assert id(_MuflonParameterSet()) == id(mpset)

    #set_log_level(INFO)
