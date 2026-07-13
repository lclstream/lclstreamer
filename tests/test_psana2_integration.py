"""Integration tests using a real XTC2 fixture file.

These tests exercise the psana2 data source classes against a tiny XTC2 file,
verifying the full code path: XTC2 -> psana2 DataSource -> lclstreamer classes -> numpy.

Requires psana2 to be installed. Skipped automatically when psana2 is unavailable.
"""

import numpy as np
import pytest
from pathlib import Path

try:
    import psana

    psana_available = True
except ImportError:
    psana_available = False

FIXTURE = Path(__file__).parent / "fixtures" / "test_fixture.xtc2"

pytestmark = [
    pytest.mark.skipif(not psana_available, reason="psana2 not installed"),
    pytest.mark.skipif(not FIXTURE.exists(), reason="XTC2 fixture not found"),
]


class TestPsana2DataSourceDirect:
    """Verify psana2 can read the fixture correctly."""

    def test_datasource_reads_events(self):
        """DataSource can open and iterate the fixture."""
        ds = psana.DataSource(files=str(FIXTURE))
        for run in ds.runs():
            events = list(run.events())
            assert len(events) == 10

    def test_detector_returns_image(self):
        """Detector returns correct shape/dtype from fixture."""
        ds = psana.DataSource(files=str(FIXTURE))
        for run in ds.runs():
            det = run.Detector("testdet")
            for event in run.events():
                image = det.raw.raw(event)
                assert image.shape == (32, 32)
                assert image.dtype == np.float32
                break


class TestPsana2TimestampIntegration:
    """Psana2Timestamp with real psana2 events."""

    def test_get_data_returns_float64_array(self):
        """Psana2Timestamp.get_data() works with real events."""
        from lclstreamer.event_data_sources.psana2.data_sources import Psana2Timestamp
        from lclstreamer.models.parameters import DataSourceParameters

        params = DataSourceParameters(type="Psana2Timestamp")
        ts = Psana2Timestamp(name="ts", parameters=params, additional_info={})

        ds = psana.DataSource(files=str(FIXTURE))
        for run in ds.runs():
            for event in run.events():
                result = ts.get_data(event)
                assert isinstance(result, np.ndarray)
                assert result.dtype == np.float64
                break


class TestPsana2DetectorInterfaceIntegration:
    """Psana2DetectorInterface with real psana2 events."""

    def test_get_data_returns_image(self):
        """Psana2DetectorInterface.get_data() works with real events."""
        from lclstreamer.event_data_sources.psana2.data_sources import (
            Psana2DetectorInterface,
        )
        from lclstreamer.models.parameters import DataSourceParameters

        ds = psana.DataSource(files=str(FIXTURE))
        for run in ds.runs():
            params = DataSourceParameters(
                type="Psana2DetectorInterface",
                psana_name="testdet",
                psana_fields="raw.raw",
            )
            iface = Psana2DetectorInterface(
                name="det",
                parameters=params,
                additional_info={"run": run},
            )

            for event in run.events():
                result = iface.get_data(event)
                assert result.shape == (32, 32)
                assert result.dtype == np.float64  # default dtype conversion
                break
