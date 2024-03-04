import dataclasses
import json
from pathlib import Path
from datetime import datetime
from enum import Enum
from json import JSONEncoder
from typing import Any


class CustomEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.timestamp()
        if isinstance(o, Enum):
            return o.value
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        return super().default(o)


def load_json(path: str | Path) -> Any:
    with open(path) as fd:
        return json.load(fd)
