"""Name-based discovery of the tactile layout inside a flattened ``observation.state``."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Must match lerobot_robot_so_flexitac.flatten_tactile_names: "tactile.<sensor>.r<RR>_c<CC>".
_TACTILE_NAME = re.compile(r"^tactile\.(?P<sensor>.+)\.r(?P<row>\d+)_c(?P<col>\d+)$")


@dataclass(frozen=True)
class TactileStateLayout:
    proprio_state_dim: int
    sensor_names: tuple[str, ...]
    tactile_shape: tuple[int, int]

    @property
    def n_tactile_sensors(self) -> int:
        return len(self.sensor_names)

    @property
    def state_dim(self) -> int:
        rows, cols = self.tactile_shape
        return self.proprio_state_dim + self.n_tactile_sensors * rows * cols

    def cli_args(self) -> str:
        """Flags to pass to ``lerobot-train`` for ``--policy.type=act_tactile``."""
        return (
            f"--policy.proprio_state_dim={self.proprio_state_dim} "
            f"--policy.n_tactile_sensors={self.n_tactile_sensors} "
            f"--policy.tactile_input_shape='[{self.tactile_shape[0]}, {self.tactile_shape[1]}]'"
        )


def _expected_names(sensor: str, shape: tuple[int, int]) -> list[str]:
    rows, cols = shape
    return [f"tactile.{sensor}.r{r:02d}_c{c:02d}" for r in range(rows) for c in range(cols)]


def infer_tactile_layout(
    state_names: Sequence[str],
    tactile_shape: tuple[int, int] = (12, 32),
) -> TactileStateLayout:
    """Derive ``proprio_state_dim`` and sensor order from ``observation.state`` feature names.

    Requires the layout produced by ``SOTactileFollower``: proprioceptive columns first,
    then each sensor's ``rows * cols`` columns contiguous and in row-major order.

    Example::

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        names = LeRobotDatasetMetadata("user/dataset").features["observation.state"]["names"]
        print(infer_tactile_layout(names).cli_args())
    """
    names = list(state_names)
    first = next((i for i, n in enumerate(names) if _TACTILE_NAME.match(n)), None)
    if first is None:
        raise ValueError("No tactile columns ('tactile.<sensor>.rRR_cCC') found in observation.state names")

    taxels = tactile_shape[0] * tactile_shape[1]
    tail = names[first:]
    if len(tail) % taxels != 0:
        raise ValueError(
            f"{len(tail)} columns follow the first tactile column; not a multiple of {taxels} "
            f"(tactile_shape={tactile_shape}). Proprioceptive columns must come before all tactile columns."
        )

    sensors: list[str] = []
    for start in range(0, len(tail), taxels):
        block = tail[start : start + taxels]
        match = _TACTILE_NAME.match(block[0])
        if match is None:
            raise ValueError(f"Expected a tactile column at state index {first + start}, got {block[0]!r}")
        sensor = match.group("sensor")
        if block != _expected_names(sensor, tactile_shape):
            raise ValueError(
                f"Tactile columns for sensor {sensor!r} are not contiguous row-major {tactile_shape} "
                f"(starting at state index {first + start})"
            )
        if sensor in sensors:
            raise ValueError(f"Sensor {sensor!r} appears twice in observation.state")
        sensors.append(sensor)

    return TactileStateLayout(proprio_state_dim=first, sensor_names=tuple(sensors), tactile_shape=tactile_shape)
