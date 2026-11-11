# lerobot-flexitac

Third-party [LeRobot](https://github.com/huggingface/lerobot) plugins for **FlexiTac** `12 × 32` array tactile sensors (via [PyFlexiTac](https://github.com/WT-MM/PyFlexiTac), 2 Mbaud USB serial). Developed for [**LeFlexiTac: Giving Robots a Sense of Touch**](https://tna001-ai.github.io/LeFlexiTac/docs.html).

They install on **stock `lerobot` with no changes to LeRobot core**. LeRobot auto-discovers them by package name.

| Package | Registers | What it does |
| :--- | :--- | :--- |
| [`lerobot_tactile_flexitac`](lerobot_tactile_flexitac/) | stream device `flexitac` | FlexiTac pad as a LeRobot stream device for **any robot**. Reversible `raw` or Viridis `heatmap` images, exact float32 access, optional multi-process backend. |
| [`lerobot_robot_so_flexitac`](lerobot_robot_so_flexitac/) | robot `so_tactile_follower` | SO-100/101 follower that stores all `12 × 32 = 384` taxels per pad **losslessly as float32** in `observation.state`, with an optional full-rate CSV sidecar. |
| [`lerobot_policy_tactile`](lerobot_policy_tactile/) | policy `act_tactile` | ACT with LeFlexiTac's 2D tactile token encoders (`TactileCNN` / `TactileAttentionCNN`). |

**Tested with:** LeRobot 0.3.4 (Python 3.11) and the latest LeRobot on PyPI (Python 3.12) in CI, plus `flexitac` 0.3.1. Unit tests use simulated hardware plus a contract test against the real `flexitac` API. Validate on your own hardware before long recordings.

---

## Installation

```bash
pip install lerobot          # if not installed yet
pip install "git+https://github.com/TNA001-AI/lerobot-flexitac.git#subdirectory=lerobot_tactile_flexitac"
pip install "git+https://github.com/TNA001-AI/lerobot-flexitac.git#subdirectory=lerobot_robot_so_flexitac"
pip install "git+https://github.com/TNA001-AI/lerobot-flexitac.git#subdirectory=lerobot_policy_tactile"
```

Install in this order: the robot package depends on the tactile package. Neither declares `lerobot` as a dependency, so installing never re-resolves your existing LeRobot / torch environment. The tactile package pulls in `flexitac`, which only needs `numpy` and `pyserial` (Python ≥ 3.11).

Find the sensor's port with `flexitac-find-port` (unplug/replug). FlexiTac boards have no USB descriptor that identifies them.

---

## Which mode should you use?

| | Stream device (`type: flexitac`) | `so_tactile_follower` robot |
| :--- | :--- | :--- |
| Robots | any | SO-100 / SO-101 |
| Stored as | `observation.images.<name>` `(H, W, 3)` uint8 | 384 float32 columns per pad in `observation.state` |
| Precision | 1/255 with `render=raw` + `--dataset.video=false` (PNG); lossy with default video encoding | exact float32 |
| Rerun live view | yes (`render=heatmap`) | no (state plot only) |
| Stock ACT / Diffusion | yes, as an extra camera | yes, but see the warnings below |
| Stock SmolVLA / Pi0 / Pi0.5 | yes, as an extra camera | **no**: these pad/truncate state to `max_state_dim=32` |
| `act_tactile` | key mode (`--policy.tactile_keys`) | state mode (recommended) |

> [!WARNING]
> **Normalization of tactile state.** Stock policies normalize `observation.state` per column with MEAN_STD. Rarely-touched taxels have std close to 0, so contacts get scaled to huge values. `act_tactile` excludes tactile columns from normalization by default (`override_tactile_normalization=true`). If you train a stock policy on `so_tactile_follower` data, expect this issue.

> [!NOTE]
> **What the values mean.** By default (`signal=normalized`) the map is `flexitac`'s `frame.normalized`. It is **scaled per frame by its own peak** once a contact exceeds `threshold`, so it shows the contact *shape*, not absolute pressure comparable across frames. This matches the original LeFlexiTac datasets. Use `signal=baseline_subtracted` for a fixed global scale.

---

## Quick start

### Any robot: live tactile heatmap in Rerun

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm \
  --robot.cameras='{tactile: {type: flexitac, port: "/dev/ttyUSB0", render: heatmap}}' \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm \
  --display_data=true
```

### SO-100/101: lossless float32 recording

```bash
lerobot-record \
  --robot.type=so_tactile_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm \
  --robot.tactile_sensors='{primary: {port: "/dev/ttyUSB0"}}' \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm \
  --dataset.repo_id=${HF_USER}/flexitac_peg_in_hole \
  --dataset.single_task="Insert peg into hole" --dataset.num_episodes=50
```

If a pad stops delivering fresh frames, the robot bridges at most `stale_timeout_s` (default 2 s) with the last valid frame and logs a warning. After that it raises instead of recording frozen tactile data.

### Train `act_tactile`

Derive the state layout from the dataset's feature names. The layout is never guessed from tensor widths:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot_policy_tactile import infer_tactile_layout

names = LeRobotDatasetMetadata("user/flexitac_peg_in_hole").features["observation.state"]["names"]
print(infer_tactile_layout(names).cli_args())
# --policy.proprio_state_dim=6 --policy.n_tactile_sensors=1 --policy.tactile_input_shape='[12, 32]'
```

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/flexitac_peg_in_hole \
  --policy.type=act_tactile \
  --policy.proprio_state_dim=6 --policy.n_tactile_sensors=1 \
  --policy.tactile_encoder_type=attention --policy.n_tactile_tokens=4
```

---

## Hardware

Printable SO-10X **Moving Jaw** and **Wrist Roll** STL files and setup steps are in [`hardware/`](hardware/INSTALL.md).

## Tests (no hardware needed)

```bash
pip install pytest opencv-python-headless
pytest lerobot_tactile_flexitac/tests lerobot_robot_so_flexitac/tests lerobot_policy_tactile/tests
```

## Citation

```bibtex
@article{tao2026leflexitac,
  author = {Naian Tao and Yifan He and Wesley Maa and Binghao Huang and Yunzhu Li},
  title = {{LeFlexiTac}: Giving Robots a Sense of Touch},
  journal = {Columbia University RoboPIL Blog},
  year = {2026},
  note = {\url{https://tna001-ai.github.io/LeFlexiTac/docs.html}},
}
```

Licensed under Apache-2.0.
