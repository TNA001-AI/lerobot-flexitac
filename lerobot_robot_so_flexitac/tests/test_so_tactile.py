"""Unit tests for lerobot_robot_so_flexitac without physical hardware."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot_robot_so_flexitac import (
    FlexiTacSensorEntryConfig,
    SOTactileFollower,
    SOTactileFollowerConfig,
    flatten_tactile_names,
    unflatten_tactile_state,
)


class _FakeTactileSensor:
    def __init__(self, shape: tuple[int, int] = (12, 32)):
        self.shape = shape
        self.is_connected = False
        self.should_fail = False
        self.data = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape) / float(
            shape[0] * shape[1]
        )

    def connect(self, warmup: bool = True) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def read_latest_tactile(self) -> np.ndarray:
        if self.should_fail:
            raise OSError("Simulated serial timeout")
        return self.data.copy()


def test_robot_config_registration_and_naming_convention():
    assert RobotConfig.get_choice_class("so_tactile_follower") is SOTactileFollowerConfig
    assert SOTactileFollowerConfig.__name__.removesuffix("Config") == SOTactileFollower.__name__


def test_flatten_and_unflatten_c_order_roundtrip():
    shape = (12, 32)
    names_joint = [f"joint_{i}.pos" for i in range(6)]
    names_tactile = flatten_tactile_names("primary", shape)
    assert len(names_tactile) == 384
    assert names_tactile[0] == "tactile.primary.r00_c00"
    assert names_tactile[-1] == "tactile.primary.r11_c31"

    all_names = names_joint + names_tactile
    rng = np.random.default_rng(7)
    joints = rng.normal(size=(4, 6)).astype(np.float32)
    tactile_maps = rng.uniform(0.0, 1.0, size=(4, 12, 32)).astype(np.float32)
    state_batch = np.concatenate([joints, tactile_maps.reshape(4, 384)], axis=-1)

    recovered = unflatten_tactile_state(all_names, state_batch, sensor_name="primary", shape=shape)
    assert recovered.shape == (4, 12, 32)
    np.testing.assert_allclose(recovered, tactile_maps, atol=1e-7)


def test_observation_features_and_fault_tolerant_get_observation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Stub out parent SOFollower __init__, connect, get_observation, disconnect so no serial arm is needed
    base_cls = SOTactileFollower.__mro__[1]
    monkeypatch.setattr(
        base_cls,
        "__init__",
        lambda self, cfg: (
            setattr(self, "config", cfg),
            setattr(self, "bus", MagicMock(motors={f"j{i}": i for i in range(6)}, is_connected=True)),
            setattr(self, "cameras", {}),
        ),
    )
    monkeypatch.setattr(base_cls, "connect", lambda self, calibrate=True: None)
    monkeypatch.setattr(
        base_cls,
        "get_observation",
        lambda self: {f"j{i}.pos": float(i) for i in range(6)},
    )
    monkeypatch.setattr(base_cls, "disconnect", lambda self: None)

    fake_sensor = _FakeTactileSensor((12, 32))
    cfg = SOTactileFollowerConfig(
        port="/dev/ttyACM0",
        tactile_sensors={
            "primary": FlexiTacSensorEntryConfig(
                port="/dev/ttyUSB0",
                record_raw_csv=True,
                raw_csv_dir=str(tmp_path / "csv"),
            )
        },
    )
    robot = SOTactileFollower(cfg, tactile_factory=lambda _n, _c: fake_sensor)

    # observation_features MUST work before connect() (as lerobot-record requires)
    assert not fake_sensor.is_connected
    ft = robot.observation_features
    assert len(ft) == 6 + 384
    assert ft["j0.pos"] is float
    assert ft["tactile.primary.r00_c00"] is float

    robot.connect(calibrate=False)
    assert fake_sensor.is_connected

    obs_ok = robot.get_observation()
    assert len(obs_ok) == 390
    state_keys = list(ft.keys())
    state_vec = np.array([obs_ok[k] for k in state_keys], dtype=np.float32)
    np.testing.assert_allclose(
        unflatten_tactile_state(state_keys, state_vec, "primary", (12, 32)),
        fake_sensor.data,
    )

    # Inject transient sensor failure: must reuse last valid frame instead of zeros!
    fake_sensor.should_fail = True
    obs_glitch = robot.get_observation()
    state_vec_glitch = np.array([obs_glitch[k] for k in state_keys], dtype=np.float32)
    np.testing.assert_allclose(state_vec_glitch, state_vec)

    # Verify sidecar CSV recorded rows
    robot.disconnect()
    assert not fake_sensor.is_connected
    csv_path = tmp_path / "csv" / "flexitac_primary.csv"
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # header + 2 observations
