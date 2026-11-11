# FlexiTac Hardware & SO-10X Gripper Setup

This directory documents the physical integration of the **FlexiTac** (`12 × 32`) array tactile sensor onto SO-100 / SO-101 / SO-10X grippers.

## 1. Custom 3D-Printed Gripper Parts

To mount the FlexiTac pad cleanly and accommodate sensor ribbon routing, replace two stock SO-10X gripper parts:

1. **Moving Jaw** — holds the `12 × 32` FlexiTac sensor pad flat against the grasping surface.
2. **Wrist Roll** — replaces the stock SO-10X wrist roll housing to accommodate the modified jaw geometry and USB/FPC cable clearance.

Download the STL files and view interactive 3D previews on the official project documentation:
- **Project Hardware Guide & STL Downloads**: https://tna001-ai.github.io/LeFlexiTac/docs.html

*(Tip: You can also drop `moving_jaw_flexitac.stl` and `wrist_roll_flexitac.stl` directly into this `hardware/` folder before pushing to GitHub.)*

## 2. Firmware Flashing & Driver Verification (`PyFlexiTac`)

FlexiTac communicates over USB serial at **`2,000,000` baud**.

1. Install `PyFlexiTac` with examples:
   ```bash
   pip install "flexitac[examples]"
   # Or from source: https://github.com/WT-MM/PyFlexiTac
   ```
2. Grant serial port permissions on Linux:
   ```bash
   sudo usermod -aG dialout $USER
   # Or temporary permission for current session:
   sudo chmod 666 /dev/ttyUSB0
   ```
3. Verify live `12 × 32` tactile streaming using `PyFlexiTac`'s built-in CLI heatmap viewer before launching `lerobot-teleoperate` or `lerobot-record`.
