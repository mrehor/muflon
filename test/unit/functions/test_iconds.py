import pytest

from muflon.functions.iconds import SimpleCppIC

def test_SimpleCppIC():
    # prepare wrapper for initial conditions
    ic = SimpleCppIC()

    # test adding of values
    N = 3
    gdim = 2
    with pytest.raises(AttributeError):
        ic.add("foo", 42.0)
    with pytest.raises(TypeError):
        ic.add("c", (42, 42))
    ic.add("c", "A*(1.0 - pow(x[0], 2.0))", A=2.0)
    ic.add("c", "B*pow(x[0], 2.0)", B=1.0)
    ic.add("v", 1.0)
    with pytest.raises(AssertionError): # one velocity component is missing
        values, coeffs = ic.get_vals_and_coeffs(N, gdim)
    ic.add("v", 2.0)

    # test unwrapping of ICs
    values, coeffs = ic.get_vals_and_coeffs(N, gdim)
    assert len(values) == len(coeffs)
    values, coeffs = ic.get_vals_and_coeffs(N, gdim, unified=True)
    assert isinstance(coeffs, dict)
