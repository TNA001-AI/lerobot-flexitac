# lerobot-flexitac

Official [LeRobot](https://github.com/huggingface/lerobot) third-party tactile plugins for **FlexiTac** (`12 × 32` array tactile sensors at `2,000,000` baud), developed as part of [**LeFlexiTac: Giving Robots a Sense of Touch**](https://tna001-ai.github.io/LeFlexiTac/docs.html).

All packages install cleanly onto **stock `lerobot`** (`>= 0.6.0`) via `pip` with **zero modifications to `lerobot` core**.

| Package | Registered Type | Description |
| :--- | :--- | :--- |
| [`lerobot_tactile_flexitac`](lerobot_tactile_flexitac/) | `flexitac` | Drop-in tactile stream plugin compatible with **any LeRobot arm**. Supports reversible `"raw"` `(H, W, 3)` tactile encoding, Viridis `"heatmap"` visualization, and optional `mp.Process` + `shared_memory` backend for multi-sensor setups. |
| [`lerobot_robot_so_flexitac`](lerobot_robot_so_flexitac/) | `so_tactile_follower` | SO-100 / SO-101 follower arm with FlexiTac fingertips, recording all `12 × 32 = 384` taxels losslessly as `float32` features in `observation.state` (plus optional full-rate CSV sidecars). |
| [`lerobot_policy_tactile`](lerobot_policy_tactile/) | `act_tactile` | Policy plugin packaging `TactileCNN`, `TactileAttentionCNN`, and `TactileTokenEncoder` so spatial tactile encoders plug directly into `lerobot-train`. |

---

## Which Mode Should You Use?

- **Universal Tactile Stream (`lerobot_tactile_flexitac`)**: Works with *any* robot (`so101_follower`, `aloha`, `koch_follower`, `ur5e`, etc.), renders live in Rerun (`--display_data=true`), and feeds straight into stock vision/VLA policies (`act`, `diffusion`, `smolvla`, `pi05`). Pair `render="raw"` with `--dataset.video=false` (lossless PNG) when exact `uint8` (`1/255`) taxel values matter.
- **Lossless `float32` Robot Plugin (`lerobot_robot_so_flexitac`)**: Stores exact `float32` normalized readings in Parquet `observation.state` without `uint8` quantization or video compression. Pair with `lerobot_policy_tactile` (`--policy.type=act_tactile`) to reshape the 384 tactile floats back into `(B, 12, 32)` grids for 2D spatial token encoding.

---

## Installation

```bash
# 1. Install stock LeRobot and the PyFlexiTac hardware driver
pip install lerobot flexitac

# 2. Install the plugins (from local checkout or PyPI)
pip install -e ./lerobot_tactile_flexitac
pip install -e ./lerobot_robot_so_flexitac
pip install -e ./lerobot_policy_tactile
```

> [!NOTE]
> Each plugin declares `dependencies = []` in `pyproject.toml` by design so installing it never re-resolves or overwrites your existing `lerobot`, `torch`, or `opencv` environment.

---

## Quick Start

### 1. Universal Tactile Stream (Any Robot + Rerun Visualization)

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras='{tactile_primary: {type: flexitac, port: "/dev/ttyUSB0", render: "heatmap", fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm \
  --display_data=true
```

### 2. Lossless `float32` Recording on SO-100 / SO-101

```bash
lerobot-record \
  --robot.type=so_tactile_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.tactile_sensors='{primary: {port: "/dev/ttyUSB0", baud_rate: 2000000}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm \
  --dataset.repo_id=${HF_USER}/flexitac_peg_in_hole \
  --dataset.single_task="Insert peg into hole" \
  --dataset.num_episodes=50
```

### 3. Policy Training

Train any stock policy directly, or use `act_tactile` from `lerobot_policy_tactile`:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/flexitac_peg_in_hole \
  --policy.type=act_tactile \
  --policy.tactile_encoder_type=attention \
  --policy.n_tactile_tokens=4
```

---

## Hardware & Gripper CAD

Printable SO-10X gripper jaws (**Moving Jaw** and **Wrist Roll**) and firmware flashing instructions are documented in [`hardware/INSTALL.md`](hardware/INSTALL.md).

## Running Tests (No Hardware Required)

```bash
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
