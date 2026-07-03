from pathlib import Path

import yaml

from lclstreamer.models.parameters import Parameters


def test_example_params():
    for path in Path("examples").iterdir():
        if not path.name.endswith(".yaml"):
            continue

        print(f"Reading example config. {path.name}")
        params = yaml.safe_load(path.read_text())
        _ = Parameters.model_validate(params)
