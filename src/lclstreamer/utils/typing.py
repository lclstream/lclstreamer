import numpy
from numpy.typing import NDArray
from typing_extensions import Any, TypeAlias

StrFloatIntNDArray: TypeAlias = NDArray[
    numpy.str_ | numpy.floating[Any] | numpy.integer[Any]
]
