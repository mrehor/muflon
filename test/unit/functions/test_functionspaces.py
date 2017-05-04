import pytest

import dolfin
from muflon.functions.functionspaces import _DiscretizationBase
from muflon.functions.functionspaces import DiscretizationMono
from muflon.functions.functionspaces import DiscretizationSemi
from muflon.common.parameters import mpset


def get_arguments():
    mesh = dolfin.UnitSquareMesh(2, 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    return (mesh, P1, P1, P2, P1, P1)

def test_DiscretizationBase():
    args = get_arguments()
    with pytest.raises(NotImplementedError):
        foo = _DiscretizationBase(*args)

@pytest.mark.parametrize("D", [DiscretizationMono, DiscretizationSemi])
def test_Discretization(D):
    args = get_arguments()
    foo = D(*args)

    # Check direct access to parameter "N" in the constructor of 'foo'
    a1 = mpset.discretization.get("N")[-2]
    a2 = foo.parameters.get("N")[-2]
    assert a2 == a1+1
    del a1, a2

    # Check inheritance of docstrings
    assert foo._split_solution_fcns.__doc__ == \
      _DiscretizationBase._split_solution_fcns.__doc__

    # Check primitive variables
    pv = foo.solution_split()
    assert len(pv) == len(args)-1
    assert pv[0].name() == "c"
    assert pv[1].name() == "mu"
