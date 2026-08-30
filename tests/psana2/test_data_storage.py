import numpy

from lclstreamer.processing_pipelines.common.data_storage import DataStorage


def test_len_is_zero_initially():
    storage = DataStorage()
    assert len(storage) == 0


def test_len_is_one_after_first_add():
    storage = DataStorage()
    storage.add_data({"x": numpy.zeros((4,), dtype=numpy.float32)})
    assert len(storage) == 1


def test_len_counts_every_add():
    storage = DataStorage()
    for _ in range(5):
        storage.add_data({"x": numpy.zeros((4,), dtype=numpy.float32)})
    assert len(storage) == 5


def test_reset_zeroes_count_and_count_resumes_from_one():
    storage = DataStorage()
    for _ in range(3):
        storage.add_data({"x": numpy.zeros((4,), dtype=numpy.float32)})

    storage.reset_data_storage()
    assert len(storage) == 0

    storage.add_data({"x": numpy.zeros((4,), dtype=numpy.float32)})
    assert len(storage) == 1


def test_len_counts_dict_typed_events():
    storage = DataStorage()
    event = {
        "detector": {
            "image": numpy.zeros((2, 2), dtype=numpy.float32),
            "mask": numpy.zeros((2, 2), dtype=numpy.float32),
        }
    }
    storage.add_data(event)
    storage.add_data(event)
    assert len(storage) == 2
