import pytest

from muflon.functions.primitives import as_primitive, PrimitiveShell

def test_PrimitiveShell():
    with pytest.raises(RuntimeError):
        as_primitive("foo")
    with pytest.raises(RuntimeError):
        PrimitiveShell(42, "answer")
