# FlexiTac hardware and SO-10X gripper setup

## 1. 3D-printed gripper parts

Replace two stock SO-10X parts so the FlexiTac pad sits flat on the jaw and its cable has clearance:

| File | Part |
| :--- | :--- |
| [`Moving_Jaw_08d.stl`](Moving_Jaw_08d.stl) | Moving jaw that holds the `12 × 32` FlexiTac pad |
| [`Wrist_Roll_08c.stl`](Wrist_Roll_08c.stl) | Wrist roll replacement matching the new jaw geometry and cable routing |

Interactive 3D previews: https://tna001-ai.github.io/LeFlexiTac/docs.html

## 2. Firmware and driver check (PyFlexiTac)

FlexiTac streams frames of `0xAA 0x55` followed by `12 × 32` uint8 values at **2,000,000 baud**.

1. Install the driver (Python ≥ 3.11):
   ```bash
   pip install "flexitac[examples]"
   ```
2. Give your user serial-port access on Linux, then log out and back in:
   ```bash
   sudo usermod -aG dialout $USER
   ```
3. Find the port. The board has no identifying USB descriptor, so use unplug/replug detection:
   ```bash
   flexitac-find-port
   ```
4. Flash the firmware and check the live heatmap with PyFlexiTac's tools (see its README) before you run `lerobot-teleoperate` or `lerobot-record`.

Keep the pad unloaded while connecting. `flexitac` calibrates a per-taxel baseline from the first `init_frames` (30) frames, unless you pass a fixed `baseline`.
