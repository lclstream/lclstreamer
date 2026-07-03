"""Generate a small XTC2 fixture file for integration testing.

Requires psana2 (lcls2) to be installed. Run once to create the fixture:

    pixi run -e psana2 python tests/fixtures/generate_xtc2_fixture.py

The generated test_fixture.xtc2 is committed to the repo so that CI
does not need psana2 just to create it.
"""

from pathlib import Path

import numpy as np
from psana.dgramedit import AlgDef, DetectorDef, DgramEdit
from psana.psexp import TransitionId

NUM_EVENTS = 10
IMAGE_SHAPE = (32, 32)
OUTPUT = Path(__file__).parent / "test_fixture.xtc2"
BUFSIZE = 1024 * 1024  # 1 MB — plenty for 32x32 float32

# Detector definition: fakecam type is recognized by psana2's detector registry.
# The data field is 'array_raw' to match the fakecam_raw_2_0_0 detector class,
# which exposes it via det.raw.raw(event).
detdef = DetectorDef("testdet", "fakecam", "det001")
algdef = AlgDef("raw", 2, 0, 0)
datadef = {"array_raw": (np.float32, 2)}

rng = np.random.default_rng(seed=42)

# Create Configure transition (defines detector names/shapes)
config = DgramEdit(transition_id=TransitionId.Configure, bufsize=BUFSIZE)
det = config.Detector(detdef, algdef, datadef)
config_buf = bytearray(BUFSIZE)
config.save(memoryview(config_buf))
config_size = config.size

with open(OUTPUT, "wb") as f:
    f.write(config_buf[:config_size])

    # BeginRun transition
    beginrun = DgramEdit(
        config_dgramedit=config, transition_id=TransitionId.BeginRun, ts=1, bufsize=BUFSIZE
    )
    br_buf = bytearray(BUFSIZE)
    beginrun.save(memoryview(br_buf))
    f.write(br_buf[:beginrun.size])

    # L1Accept transitions (actual events)
    for i in range(NUM_EVENTS):
        l1 = DgramEdit(
            config_dgramedit=config,
            transition_id=TransitionId.L1Accept,
            ts=i + 2,
            bufsize=BUFSIZE,
        )
        det.raw.array_raw = rng.random(IMAGE_SHAPE, dtype=np.float32)
        l1.adddata(det.raw)
        l1_buf = bytearray(BUFSIZE)
        l1.save(memoryview(l1_buf))
        f.write(l1_buf[:l1.size])

print(f"Generated {OUTPUT} ({OUTPUT.stat().st_size} bytes, {NUM_EVENTS} events)")
