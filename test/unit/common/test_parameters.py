import pytest

from dolfin import mpi_comm_world, set_log_level, DEBUG
from muflon.common.parameters import mpset, _MuflonParameterSet

import os

def test_mpset(tmpdir):
    set_log_level(DEBUG)

    # Print parameters and their values
    mpset.show()

    # Try to add parameter
    mpset.add("foo", "bar")
    foo_info = mpset.get("foo") # 1st access
    assert foo_info[-1] == 0 # check 'change' value
    assert foo_info[-2] == 1 # check 'access' value

    # Try direct access to a parameter
    mpset["foo"] = "bar_" # 1st change
    foo_info = mpset.get("foo") # 2nd access
    assert foo_info[-1] == 1 # check 'change' value
    assert foo_info[-2] == 2 # check 'access' value

    # Try to write parameters to a file
    mpset.write(mpi_comm_world(), str(tmpdir)+"/foo.xml") # 3rd access
    foo_info = mpset.get("foo") # 4th access
    assert foo_info[-1] == 1 # check 'change' value
    assert foo_info[-2] == 4 # check 'access' value

    # Try to read parameters back
    mpset.read(str(tmpdir)+"/foo.xml") # 2nd change
    foo_info = mpset.get("foo") # 5th access
    assert foo_info[-1] == 2 # check 'change' value
    assert foo_info[-2] == 5 # check 'access' value

    # Check that every other call points to the same object
    assert id(_MuflonParameterSet()) == id(mpset)
