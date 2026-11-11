# lerobot_policy_tactile

Third-party policy plugin (`lerobot_policy_*`) packaging **LeFlexiTac** spatial tactile encoders (`TactileCNN`, `TactileAttentionCNN`, `TactileTokenEncoder`) and the `act_tactile` policy for stock [LeRobot](https://github.com/huggingface/lerobot).

## Naming Convention (LeRobot `factory.py` Contract)

Stock `lerobot-train` resolves third-party policies registered via `@PreTrainedConfig.register_subclass("<type>")` using three sibling modules:

- `configuration_act_tactile.py` — `ACTTactileConfig` (`@PreTrainedConfig.register_subclass("act_tactile")`)
- `modeling_act_tactile.py` — `ACTTactilePolicy`
- `processor_act_tactile.py` — `make_act_tactile_pre_post_processors`

*(You can add `smolvla_tactile`, `diffusion_tactile`, or `pi05_tactile` in this exact same package following the same `configuration_*` / `modeling_*` / `processor_*` triplet!)*

## Supported Tactile Dataset Layouts

`extract_tactile_maps_from_batch()` automatically supports:
1. **Lossless `float32` state slices** recorded by `lerobot_robot_so_flexitac` (`observation.state[..., proprio_dim:]` reshaped to `(B, n_sensors, 12, 32)`).
2. **Encoded tactile image streams** recorded by `lerobot_tactile_flexitac` (`observation.images.tactile_*`).
3. **Legacy `observation.tactile.*` tensors** from the original `LeFlexiTac` research datasets.
