import pytest

from muflon.functions.primitives import as_primitive, PrimitiveShell

def test_PrimitiveShell():
    with pytest.raises(TypeError):
        as_primitive("foo")
    with pytest.raises(TypeError):
        PrimitiveShell(42, "answer")
