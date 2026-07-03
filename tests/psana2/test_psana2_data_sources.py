"""
Mock-based unit tests for psana2 data source classes.

These tests exercise Psana2Timestamp, Psana2DetectorInterface, and
Psana2RunInfo without requiring psana to be installed.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy
import pytest

from lclstreamer.event_data_sources.psana2.data_sources import (
    Psana2DetectorInterface,
    Psana2RunInfo,
    Psana2Timestamp,
)
from lclstreamer.models.parameters import (
    Psana2TimestampParameters,
    Psana2DetectorInterfaceParameters,
    Psana2RunInfoParameters,
)


# -- Helpers --


def _make_run(detector_mock=None, expt="mfxl1038923", timestamp=1728946748, runnum=278):
    """Create a mock psana2 run object."""
    run = MagicMock()
    run.expt = expt
    run.timestamp = timestamp
    run.runnum = runnum
    if detector_mock is not None:
        run.Detector.return_value = detector_mock
    return run


# -- Psana2Timestamp tests --


class TestPsana2Timestamp:
    def setup_method(self):
        params = Psana2TimestampParameters(type="Psana2Timestamp")
        self._name = "timestamp"
        self.ts = Psana2Timestamp(
            name=self._name, parameters=params, additional_info={}
        )

    def test_basic_timestamp(self):
        event = MagicMock()
        event.timestamp = 1728946748.123456789
        result = self.ts.get_data(event)
        assert isinstance(result, dict)
        data = result[self._name]
        assert isinstance(data, numpy.ndarray)
        assert data.dtype == numpy.float64
        assert float(data) == pytest.approx(1728946748.123456789)

    def test_zero_timestamp(self):
        event = MagicMock()
        event.timestamp = 0.0
        result = self.ts.get_data(event)
        data = result[self._name]
        assert float(data) == 0.0

    def test_integer_timestamp(self):
        """psana2 timestamps can be integer-valued."""
        event = MagicMock()
        event.timestamp = 1728946748
        result = self.ts.get_data(event)
        data = result[self._name]
        assert data.dtype == numpy.float64
        assert float(data) == 1728946748.0


# -- Psana2DetectorInterface tests --


class TestPsana2DetectorInterface:
    def _make_interface(self, mock_detector, **extra):
        """Create a Psana2DetectorInterface with a mocked run.Detector."""
        run = _make_run(detector_mock=mock_detector)
        params = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface", **extra)
        self._name = "test_det"
        return Psana2DetectorInterface(
            name=self._name,
            parameters=params,
            additional_info={"run": run},
        )

    def _make_callable(self, return_this):
        def f(event=None):
            return return_this
        return f

    def test_single_callable_field(self):
        """psana_fields: 'raw' → det.raw(event)"""
        mock_det = MagicMock()
        mock_det.raw = self._make_callable(numpy.ones((16, 352, 384)))

        iface = self._make_interface(
            mock_det, psana_name="epix10k2M", psana_fields="raw"
        )
        event = MagicMock()
        result = iface.get_data(event)[self._name]

        assert result.shape == (16, 352, 384)

    def test_dotted_field_traversal(self):
        """psana_fields: 'raw.image' → det.raw.image(event)"""
        image_data = numpy.random.rand(1667, 1668).astype(numpy.float64)
        mock_det = MagicMock()
        mock_det.raw.image = self._make_callable(image_data)

        iface = self._make_interface(
            mock_det, psana_name="epix10k2M", psana_fields="raw.image"
        )
        event = MagicMock()
        result = iface.get_data(event)[self._name]

        numpy.testing.assert_array_equal(result, image_data)

    def test_non_callable_field(self):
        """When the traversed attribute is not callable, return it directly."""
        mock_det = MagicMock()
        mock_det.raw.some_value = numpy.array([42.0])
        # Make some_value not callable
        type(mock_det.raw).some_value = property(lambda self: numpy.array([42.0]))

        iface = self._make_interface(
            mock_det, psana_name="det", psana_fields="raw.some_value"
        )
        event = MagicMock()
        result = iface.get_data(event)[self._name]
        assert result.dtype == numpy.float64

    def test_pv_mode(self):
        """psana_name contains ':' → PV mode, det(event) called directly."""
        mock_det = MagicMock()
        mock_det.return_value = 42.0

        iface = self._make_interface(mock_det, psana_name="ABC:DEF:GHI")
        event = MagicMock()
        result = iface.get_data(event)[self._name]

        mock_det.assert_called_once_with(event)
        assert float(result) == 42.0

    def test_type_error_fallback(self):
        """When base(event) raises TypeError, falls back to base()."""
        mock_det = MagicMock()
        fallback_value = numpy.array([1.0, 2.0, 3.0])

        def raw():
            return fallback_value
        mock_det.raw.raw = raw

        iface = self._make_interface(
            mock_det, psana_name="det", psana_fields="raw.raw"
        )
        event = MagicMock()
        result = iface.get_data(event)[self._name]
        numpy.testing.assert_array_equal(result, fallback_value)

    def test_custom_dtype(self):
        """dtype parameter is applied to the output array."""
        mock_det = MagicMock()
        def calib():
            return numpy.ones((10,))
        mock_det.calib = calib

        iface = self._make_interface(
            mock_det,
            psana_name="det",
            psana_fields="calib",
            dtype="float32",
        )
        event = MagicMock()
        result = iface.get_data(event)[self._name]
        assert result.dtype == numpy.float32

    def test_detector_created_from_run(self):
        """Detector is created via run.Detector(psana_name)."""
        mock_det = MagicMock()

        def raw():
            return numpy.ones((10,))
        mock_det.raw = raw

        run = _make_run(detector_mock=mock_det)
        params = Psana2DetectorInterfaceParameters(type="Psana2DetectorInterface",
                                                   psana_name="mydetector",
                                                   psana_fields="raw",
                                                  )
        Psana2DetectorInterface(name="test", parameters=params, additional_info={"run": run})

        run.Detector.assert_called_once_with("mydetector")


# -- Psana2RunInfo tests --


class TestPsana2RunInfo:
    def test_returns_run_metadata(self):
        run = _make_run(expt="mfxl1038923", timestamp=1728946748, runnum=278)
        params = Psana2RunInfoParameters(type="Psana2RunInfo")
        name  = "run_info"
        info = Psana2RunInfo(
            name=name,
            parameters=params,
            additional_info={"run": run, "source_identifier": "exp=mfxl1038923:run=278"},
        )

        event = MagicMock()  # should be ignored
        result = info.get_data(event)
        expected = {
            name + "_experiment": "mfxl1038923",
            name + "_run_timestamp": "1728946748",
            name + "_run_number": "278",
            name + "_source_identifier": "exp=mfxl1038923:run=278",
        }

        assert all(value.dtype.kind == "U" for value in result.values())  # Unicode string dtype
        assert result.keys() == expected.keys()
        for key, value in expected.items():
            assert result[key].item() == value

    def test_event_is_ignored(self):
        """get_data returns the same cached data regardless of event."""
        run = _make_run()
        params = Psana2RunInfoParameters(type="Psana2RunInfo")
        info = Psana2RunInfo(
            name="run_info",
            parameters=params,
            additional_info={"run": run, "source_identifier": "test"},
        )

        result1 = info.get_data(MagicMock())
        result2 = info.get_data(MagicMock())
        numpy.testing.assert_array_equal(result1, result2)
