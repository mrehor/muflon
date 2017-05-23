import pytest

import dolfin

from muflon.forms.incompressible import FormsICS
from muflon.functions.discretization import DiscretizationFactory

from unit.functions.test_discretization import get_arguments

@pytest.mark.parametrize("scheme", ["Monolithic",]) # "SemiDecoupled", "FullyDecoupled"
def test_FormsICS(scheme):
    args = get_arguments(2)
    DS = DiscretizationFactory.create(scheme, *args)
    forms = FormsICS(DS)
