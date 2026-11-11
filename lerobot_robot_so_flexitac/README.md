# lerobot_robot_so_flexitac

SO-100 / SO-101 / SO-10X follower robot plugin (`so_tactile_follower`) with **lossless `float32` FlexiTac tactile recording** for stock [LeRobot](https://github.com/huggingface/lerobot).

## How It Works

Stock `lerobot` stores scalar `float` observation features losslessly in Parquet `observation.state`. `SOTactileFollower` flattens each `(12, 32)` FlexiTac pad in row-major (`np.ravel`) order into `384` named `float` columns:

```text
tactile.<sensor_name>.r00_c00 ... tactile.<sensor_name>.r11_c31
```

Downstream policies (`lerobot_policy_tactile`) or analysis scripts reconstruct the exact `(12, 32)` grid via `unflatten_tactile_state()` or `.view(-1, 12, 32)`.

### Key Design Highlights

1. **Auto-Discovery Bridge**: Importing `lerobot_robot_so_flexitac` (which stock `lerobot` auto-discovers via `lerobot_robot_*`) also automatically imports `lerobot_tactile_flexitac`, registering the `flexitac` tactile stream type globally.
2. **Fault-Tolerant Contact Continuity**: Transient serial read glitches reuse the last valid tactile map (`_last_valid_frames`) rather than injecting `0.0` (which physically mimics sudden loss of contact).
3. **Optional High-Rate Raw CSV Sidecar (`record_raw_csv=True`)**: Logs full-rate `float32` tactile frames stamped with `epoch_ns` alongside `lerobot-record`.
