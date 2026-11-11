# lerobot_policy_tactile

LeRobot policy plugin with **LeFlexiTac** 2D tactile encoders (`TactileCNN`, `TactileAttentionCNN`, `TactileTokenEncoder`) and the `act_tactile` policy.

```bash
pip install "git+https://github.com/TNA001-AI/lerobot-flexitac.git#subdirectory=lerobot-policy-tactile"
```

## How `act_tactile` uses tactile data

- **State mode** (default). Tactile is the contiguous tail of `observation.state` recorded by `so_tactile_follower`.
  - Set `proprio_state_dim` and `n_tactile_sensors` from `infer_tactile_layout(names).cli_args()`. A mismatched layout raises a clear error; nothing is guessed from tensor widths.
  - The ACT state token sees only the proprioceptive part. Each pad is reshaped to `(12, 32)` and turned into `n_tactile_tokens` tokens.
  - Per-column state normalization would divide rarely-touched taxels by a near-zero std. The model inverts LeRobot's normalization for tactile columns (MEAN_STD, MIN_MAX, QUANTILES, QUANTILE10) using dataset stats stored as checkpoint buffers, so the encoder always sees raw `[0, 1]` taxels. Proprioception stays normalized. This doesn't depend on which processor a LeRobot version picks; 0.4.x uses ACT's processors for ACT subclasses.
- **Key mode** (`--policy.tactile_keys='["observation.images.tactile"]'`). Reads tactile tensors or `flexitac` image streams. Listed image keys go only to the tactile encoder, not the ResNet backbone. Image streams arrive already normalized by the image preprocessor.

The ACT backbone is built once: the tactile model is substituted at construction time, so no throwaway ACT/ResNet is created.

## Plugin layout

`lerobot-train` resolves third-party policies by sibling module names:

- `configuration_act_tactile.py`: `ACTTactileConfig`, registered as `act_tactile`
- `modeling_act_tactile.py`: `ACTTactilePolicy`
- `processor_act_tactile.py`: `make_act_tactile_pre_post_processors`

Add `smolvla_tactile`, `diffusion_tactile`, etc. with the same three-file pattern.
