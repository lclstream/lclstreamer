"""
Mock-based unit tests for psana1 data source classes.

These tests exercise Psana1Timestamp and Psana1DetectorInterface without
requiring psana to be installed. All psana objects are replaced with mocks.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy
import pytest

# Create mock psana module before importing data_sources
mock_psana = MagicMock()
sys.modules["psana"] = mock_psana

from lclstreamer.event_data_sources.psana1.data_sources import (
    Psana1DetectorInterface,
    Psana1Timestamp,
)
from lclstreamer.models.parameters import DataSourceParameters


# -- Helpers --


def _make_event(seconds: int, nanoseconds: int) -> MagicMock:
    """Create a mock psana1 event with timestamp."""
    event_id = MagicMock()
    event_id.time.return_value = (seconds, nanoseconds)

    event = MagicMock()
    event.get.return_value = event_id
    return event


def _make_params(**extra) -> DataSourceParameters:
    """Create DataSourceParameters with extra fields."""
    return DataSourceParameters(type="Psana1DetectorInterface", **extra)


# -- Psana1Timestamp tests --


class TestPsana1Timestamp:
    def setup_method(self):
        params = DataSourceParameters(type="Psana1Timestamp")
        self.ts = Psana1Timestamp(
            name="timestamp", parameters=params, additional_info={}
        )

    def test_basic_timestamp(self):
        event = _make_event(1728946748, 123456789)
        result = self.ts.get_data(event)
        assert isinstance(result, numpy.ndarray)
        # On npz-serializer branch: string concatenation "seconds.nanoseconds"
        expected = numpy.array("1728946748.123456789")
        assert str(result) == str(expected)

    def test_zero_timestamp(self):
        event = _make_event(0, 0)
        result = self.ts.get_data(event)
        expected = numpy.array("0.0")
        assert str(result) == str(expected)

    def test_max_nanoseconds(self):
        event = _make_event(1728946748, 999999999)
        result = self.ts.get_data(event)
        expected = numpy.array("1728946748.999999999")
        assert str(result) == str(expected)

    def test_event_get_called_with_event_id(self):
        """Verify event.get() is called with the EventId class."""
        event = _make_event(100, 200)
        self.ts.get_data(event)
        event.get.assert_called_once()


# -- Psana1DetectorInterface tests --


class TestPsana1DetectorInterface:
    def _make_interface(self, mock_detector, **extra):
        """Create a Psana1DetectorInterface with a mocked Detector."""
        with patch(
            "lclstreamer.event_data_sources.psana1.data_sources.Detector",
            return_value=mock_detector,
        ):
            params = _make_params(**extra)
            return Psana1DetectorInterface(
                name="test_det", parameters=params, additional_info={}
            )

    def test_single_callable_field(self):
        """psana_fields: 'calib' → det.calib(event)"""
        mock_det = MagicMock()
        mock_det.calib.return_value = numpy.ones((16, 352, 384))

        iface = self._make_interface(
            mock_det, psana_name="epix10k2M", psana_fields="calib"
        )
        event = MagicMock()
        result = iface.get_data(event)

        mock_det.calib.assert_called_once_with(event)
        assert result.shape == (16, 352, 384)

    def test_dotted_field_traversal(self):
        """psana_fields: 'raw.image' → det.raw.image(event)"""
        image_data = numpy.random.rand(1667, 1668).astype(numpy.float64)
        mock_det = MagicMock()
        mock_det.raw.image.return_value = image_data

        iface = self._make_interface(
            mock_det, psana_name="epix10k2M", psana_fields="raw.image"
        )
        event = MagicMock()
        result = iface.get_data(event)

        mock_det.raw.image.assert_called_once_with(event)
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
        result = iface.get_data(event)
        assert result.dtype == numpy.float64

    @pytest.mark.xfail(
        reason="Known bug: 'param' is unbound in PV mode (line 158)",
        raises=UnboundLocalError,
    )
    def test_pv_mode(self):
        """psana_name contains ':' → PV mode, det(event) called directly."""
        mock_det = MagicMock()
        mock_det.return_value = 42.0

        iface = self._make_interface(mock_det, psana_name="ABC:DEF:GHI")
        event = MagicMock()
        result = iface.get_data(event)

        mock_det.assert_called_once_with(event)
        assert float(result) == 42.0

    def test_event_codes_padding(self):
        """eventCodes field gets zero-padded to length 256."""
        codes = numpy.array([140, 141, 142, 162])
        mock_det = MagicMock()
        mock_det.eventCodes.return_value = codes

        iface = self._make_interface(
            mock_det, psana_name="EvrData", psana_fields="eventCodes"
        )
        event = MagicMock()
        result = iface.get_data(event)

        assert len(result) == 256
        numpy.testing.assert_array_equal(result[:4], codes)
        assert numpy.all(result[4:] == 0)

    def test_type_error_fallback(self):
        """When base(event) raises TypeError, falls back to base()."""
        mock_det = MagicMock()
        fallback_value = numpy.array([1.0, 2.0, 3.0])
        mock_method = MagicMock(side_effect=[TypeError("no args"), fallback_value])
        mock_det.raw.info = mock_method

        iface = self._make_interface(
            mock_det, psana_name="det", psana_fields="raw.info"
        )
        event = MagicMock()
        result = iface.get_data(event)
        numpy.testing.assert_array_equal(result, fallback_value)

    def test_custom_dtype(self):
        """dtype parameter is applied to the output array."""
        mock_det = MagicMock()
        mock_det.calib.return_value = numpy.ones((10,))

        iface = self._make_interface(
            mock_det,
            psana_name="det",
            psana_fields="calib",
            dtype=numpy.float32,
        )
        event = MagicMock()
        result = iface.get_data(event)
        assert result.dtype == numpy.float32

    def test_default_dtype_is_float64(self):
        """Default dtype should be float64."""
        mock_det = MagicMock()
        mock_det.calib.return_value = numpy.ones((10,))

        iface = self._make_interface(
            mock_det, psana_name="det", psana_fields="calib"
        )
        event = MagicMock()
        result = iface.get_data(event)
        assert result.dtype == numpy.float64

    def test_multiple_fields(self):
        """Multiple psana_fields returns concatenated array."""
        mock_det = MagicMock()
        mock_det.raw.return_value = numpy.array([1.0])
        mock_det.fex.return_value = numpy.array([2.0])

        iface = self._make_interface(
            mock_det, psana_name="det", psana_fields=["raw", "fex"]
        )
        event = MagicMock()
        result = iface.get_data(event)
        # Multiple fields returns a list of arrays → shape (2, 1)
        assert result.shape == (2, 1)
        numpy.testing.assert_array_equal(result.flatten(), [1.0, 2.0])
