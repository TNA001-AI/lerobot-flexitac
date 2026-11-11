# lerobot_robot_so_flexitac

SO-100 / SO-101 follower (`so_tactile_follower`) that records FlexiTac pads **losslessly as float32** in `observation.state`. Works with stock [LeRobot](https://github.com/huggingface/lerobot) 0.3.x and the latest release.

```bash
pip install "git+https://github.com/TNA001-AI/lerobot-flexitac.git#subdirectory=lerobot_tactile_flexitac"
pip install "git+https://github.com/TNA001-AI/lerobot-flexitac.git#subdirectory=lerobot_robot_so_flexitac"
```

## Data layout

Each pad becomes `rows × cols` named float features, in `np.ravel` (row-major) order, placed after the joint positions:

```text
shoulder_pan.pos ... gripper.pos, tactile.<sensor>.r00_c00 ... tactile.<sensor>.r11_c31
```

Recover a pad with `unflatten_tactile_state(names, state, "<sensor>")`, which selects columns by name, not position. Policies get the layout from `lerobot_policy_tactile.infer_tactile_layout(names)`.

## Per-sensor options (`--robot.tactile_sensors='{name: {...}}'`)

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `port`, `baud_rate`, `shape` | `/dev/ttyUSB0`, `2000000`, `[12, 32]` | Serial connection and pad size |
| `signal` | `normalized` | `normalized` (flexitac per-frame peak scaling) or `baseline_subtracted` (fixed scale) |
| `max_age_ms` | `200` | Frames older than this count as stale for an observation |
| `stale_timeout_s` | `2.0` | How long stale data is bridged with the last valid frame before `get_observation()` raises |
| `process_backend` | `false` | Read the serial port in a separate process (multi-pad setups) |
| `record_raw_csv`, `raw_csv_dir`, `csv_flush_every` | `false`, `~/.cache/lerobot_flexitac/raw_sessions/<ts>`, `100` | Full-rate CSV of every sensor frame (`epoch_ns`, `seq`, taxels) |

## Lifecycle guarantees

- `observation_features` is available before `connect()`, which matches the order `lerobot-record` uses.
- If a tactile pad fails to connect, the arm is disconnected again and the error is re-raised.
- `is_connected` is true only if the arm **and** every pad are connected.
- A shape mismatch raises immediately, because silently misaligned state columns would corrupt the dataset.
