from dataclasses import dataclass, field
from typing import Any

import numpy
from numpy.typing import DTypeLike

from ...utils.logging import log_error_and_exit
from ...utils.typing import StrFloatIntNDArray


@dataclass
class DataContainer:
    """
    Dataclass used to store accumulated numpy arrays

    Attributes:

        data: A list of numpy arrays accumulated so far for this data source

        dtype: The numpy dtype of the arrays, inferred from the first array added

        shape: The shape of each individual array, inferred from the first array added
    """

    data: list[StrFloatIntNDArray] = field(default_factory=list)
    #data: dict[str, StrFloatIntNDArray | dict [str, StrFloatIntNDArray | None] | None] = field(default_factory=dict)
    dtype: DTypeLike | None = None
    shape: tuple[int, ...] | None = None


class DataStorage:
    """
    See documentation of the `__init__` function
    """

    def __init__(self) -> None:
        """
        Initializes a Data Storage object

        Data Storage objects are containers that can store numpy arrays and allow
        bulk retrieval of the stored data
        """

        self._data_containers: dict[str, DataContainer | dict[str, DataContainer]] = {}
        # The data labels each data source provides, to null-fill a missing source
        self._source_subkeys: dict[str, list[str]] = {}
        self._count: int = 0

    def __len__(self) -> int:
        """
        Returns the number of data entries currently stored

        Returns:

            count: The number of times `add_data` has been called since the last
                reset
        """
        return self._count

    def add_data(self, data: dict[str, StrFloatIntNDArray | None]) -> None:
        """
        Adds data to the Data Storage object

        The function takes a dictionary storing numpy arrays, each identified
        by a dictionary key label. When called for the first time, it uses
        the incoming data to determine labels and dtypes of the numpy arrays to
        accumulate. Subsequent calls accept data arrays with the same dtypes as earlier
        ones, or data whose value is None. A label first seen after the initial call
        is added, with the earlier events null-filled. If the data value is None, this
        function fills the missing data with appropriate null values (numpy.NaN for
        float data, the number -999 for signed int data, the largest value for
        unsigned int data, and the string "None" for str data)

        Arguments:

            data: a dictionary storing numpy arrays
        """
        if len(self._data_containers) == 0:
            data_source_name: str
            for data_source_name in data:
                data_value: Any | None = data[data_source_name]
                if data_value is None:
                    # Cannot be sized yet: its container is created on a later event
                    continue
                elif isinstance(data_value, dict):
                    data_container: DataContainer
                    self._source_subkeys[data_source_name] = []
                    for sub_data_name, sub_data in data_value.items():
                        if sub_data is None:
                            continue
                        subdata_container = DataContainer(
                            data=[sub_data],
                            dtype=sub_data.dtype,
                            shape=sub_data.shape,
                        )
                        self._data_containers[sub_data_name] = subdata_container
                        self._source_subkeys[data_source_name].append(sub_data_name)
                else:
                    data_container = DataContainer(
                        data=[data_value],
                        dtype=data_value.dtype,
                        shape=data_value.shape,
                    )
                    self._data_containers[data_source_name] = data_container
                    self._source_subkeys[data_source_name] = [data_source_name]
        else:
            for data_name, subdata in data.items():
                dataitems = subdata.items() if isinstance(subdata, dict) else [(data_name, subdata)]
                if subdata is None:
                    dataitems = [
                        (sub_data_name, None)
                        for sub_data_name in self._source_subkeys.get(data_name, [])
                    ]
                for data_source_name, data_value in dataitems:
                    if data_source_name not in self._data_containers:
                        if data_value is None:
                            continue
                        # First reading of a label: null-fill the earlier events
                        self._data_containers[data_source_name] = DataContainer(
                            data=[
                                self._null_value(data_value.shape, data_value.dtype)
                                for _ in range(self._count)
                            ],
                            dtype=data_value.dtype,
                            shape=data_value.shape,
                        )
                        subkeys: list[str] = self._source_subkeys.setdefault(data_name, [])
                        if data_source_name not in subkeys:
                            subkeys.append(data_source_name)
                    data_container = self._data_containers[data_source_name]

                    if data_value is None:
                        if data_container.shape is not None:
                            data_container.data.append(
                                self._null_value(data_container.shape, data_container.dtype)
                            )
                            continue
                    else:
                        if data_value.dtype != data_container.dtype:
                            log_error_and_exit(
                                f"The dtype of the data entry {data_source_name} in the "
                                "current event does not match the dtype of the data "
                                "with which this label was originally initialized"
                            )
                        if data_value.shape != data_container.shape:
                            log_error_and_exit(
                                f"The shape of the data entry {data_source_name} in the "
                                "current event does not match the shape of the data "
                                "with which this label was originally initialized"
                            )
                        data_container.data.append(data_value)
        self._count += 1

    def _null_value(
        self, shape: tuple[int, ...], dtype: DTypeLike
    ) -> StrFloatIntNDArray:
        """A null placeholder: NaN for float, -999 for signed int, the largest value for
        unsigned int, "None" otherwise"""
        if numpy.issubdtype(dtype, numpy.floating):
            return numpy.full(shape, numpy.float64("nan"), dtype=dtype)
        elif numpy.issubdtype(dtype, numpy.signedinteger):
            return numpy.full(shape, -999, dtype=dtype)
        elif numpy.issubdtype(dtype, numpy.unsignedinteger):
            return numpy.full(shape, numpy.iinfo(dtype).max, dtype=dtype)
        return numpy.full(shape, "None")

    def retrieve_stored_data(self) -> dict[str, StrFloatIntNDArray | None]:
        """
        Retuns the data stored in the Data Storage container object

        The data is returned as dictionary of numpy arrays. The keys of the
        dictionary match the labels of the stored data. The array associated
        with each label stores the accumulated data, with the fist axis
        representing each subsequent data item added, and the rest of the axes
        representing the accumulated data

        Returns:

            stored_data: A dictionary containing the data accumulated by the
                Data Storage container
        """

        stored_data: dict[str, StrFloatIntNDArray | None] = {}

        data_source_name: str
        for data_source_name in self._data_containers:
            stored_data[data_source_name] = numpy.stack(
                self._data_containers[data_source_name].data,
            )

        return stored_data

    def reset_data_storage(self) -> None:
        """
        Resets the Data Storage container

        Clears all accumulated arrays from every data container and resets the
        internal event counter to zero. The container labels and dtypes inferred
        from the first event are preserved so that the storage can be reused for
        a new batch without re-initialization
        """
        data_source_name: str
        for data_source_name in self._data_containers:
            self._data_containers[data_source_name].data = []
        self._count = 0


def is_null_value(value: Any) -> bool:
    """Whether a value consists only of the null placeholders of DataStorage._null_value:
    NaN for float, -999 for signed int, the largest value for unsigned int"""
    array: Any = numpy.asarray(value)
    if numpy.issubdtype(array.dtype, numpy.floating):
        return not bool(numpy.isfinite(array).any())
    elif numpy.issubdtype(array.dtype, numpy.signedinteger):
        return bool((array == -999).all())
    elif numpy.issubdtype(array.dtype, numpy.unsignedinteger):
        return bool((array == numpy.iinfo(array.dtype).max).all())
    return False
