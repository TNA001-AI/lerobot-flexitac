# lerobot_tactile_flexitac

Drop-in tactile sensor plugin for **FlexiTac** (`12 × 32` piezoresistive tactile pads at `2,000,000` baud) in [LeRobot](https://github.com/huggingface/lerobot).

Registers the `flexitac` stream sensor type (`FlexiTacTactileConfig` / `FlexiTacTactile`).

## Features

- **Dual Encoding Modes (`render`)**:
  - `render="raw"` *(default)*: Reversibly maps `[0, 1]` tactile floats onto 3-channel `[0, 255]` `uint8` arrays (`12 × 32` native or nearest-neighbor upscaled). Pair with `--dataset.video=false` (lossless PNG) to recover per-taxel values within `1/255` quantization via `decode_tactile_image()`.
  - `render="heatmap"`: Viridis colormap with configurable EMA temporal smoothing (`temporal_alpha`) for Rerun `--display_data=true` and human demos.
- **Native `float32` Access**: `read_latest_tactile()` exposes the exact `(12, 32)` `float32` tactile map directly for robot plugins (`lerobot_robot_so_flexitac`).
- **Fault-Tolerant Buffering**: Transient serial read glitches reuse `_last_valid_tactile` rather than injecting spurious zero frames (which downstream policies could misinterpret as sudden loss of contact).
- **Optional Multiprocessing + Shared Memory (`process_backend=True`)**: Runs each sensor's `2,000,000` baud serial reader in an isolated `spawn` worker process writing to `multiprocessing.shared_memory` for zero-copy dual-sensor setups.

## Installation

```bash
pip install flexitac
pip install -e .
```
