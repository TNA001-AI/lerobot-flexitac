# lerobot_tactile_flexitac

FlexiTac (`12 × 32`, 2 Mbaud) tactile pad as a [LeRobot](https://github.com/huggingface/lerobot) stream device. Registers type `flexitac` (`FlexiTacTactileConfig` / `FlexiTacTactile`).

```bash
pip install "git+https://github.com/TNA001-AI/lerobot-flexitac.git#subdirectory=lerobot_tactile_flexitac"
```

## Behavior

- **One owner of the serial port.** A reader thread (or, with `process_backend=True`, a spawned process writing to shared memory) is the only code that touches the port. All read APIs serve the latest ingested frame.
- **Two access paths.**
  - `read_latest_tactile(max_age_ms)` returns the exact `(12, 32)` float32 map.
  - `read()`, `async_read()` and `read_latest()` return `(H, W, 3)` uint8 for stock `lerobot-record` and Rerun.
- **No silent freezes.**
  - `read_latest*` raises `TimeoutError` when the latest frame is older than `max_age_ms`.
  - `async_read` raises `TimeoutError` instead of repeating an old frame.
  - `last_error` holds the most recent reader error.
- **`signal`.**
  - `normalized` (default) is flexitac's per-frame peak-normalized map.
  - `baseline_subtracted` uses a fixed global scale and keeps absolute pressure comparable across frames.
- **`render`.**
  - `raw` (default) is a reversible gray encoding at native `12 × 32` (nearest-neighbor if resized). Invert it with `decode_tactile_image()`, and record with `--dataset.video=false` to keep it lossless.
  - `heatmap` is Viridis, smoothed with an EMA (`temporal_alpha`) that advances once per sensor frame, regardless of how often it's read.
- **Frame listeners.** `add_frame_listener(fn(map, epoch_ns, seq))` is called for every sensor frame. It is used for full-rate logging.
- **`find_cameras()`** lists every serial port, because FlexiTac boards can't be fingerprinted. Use `flexitac-find-port` to identify yours.
