"""Tests for tactile encoders, layout discovery, normalization override and act_tactile."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot_policy_tactile import (
    TactileTokenEncoder,
    extract_tactile_maps_from_batch,
    infer_tactile_layout,
    override_tactile_stats,
)

JOINTS = [f"{j}.pos" for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


def _tactile_names(sensor: str) -> list[str]:
    return [f"tactile.{sensor}.r{r:02d}_c{c:02d}" for r in range(12) for c in range(32)]


@pytest.mark.parametrize("encoder_type", ["cnn", "attention"])
@pytest.mark.parametrize("n_tokens", [1, 4])
def test_tactile_token_encoder_forward_and_gradients(encoder_type: str, n_tokens: int):
    enc = TactileTokenEncoder(encoder_type=encoder_type, input_shape=(12, 32), feature_dim=256, n_tokens=n_tokens)
    x = torch.rand(4, 12, 32, requires_grad=True)
    tokens = enc(x)
    assert tokens.shape == (4, n_tokens, 256)
    tokens.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_infer_tactile_layout_from_feature_names():
    layout = infer_tactile_layout(JOINTS + _tactile_names("left") + _tactile_names("right"))
    assert layout.proprio_state_dim == 6
    assert layout.sensor_names == ("left", "right")
    assert layout.state_dim == 6 + 2 * 384
    assert "--policy.n_tactile_sensors=2" in layout.cli_args()

    # Bimanual (12 joints) is handled by names, not by guessing from tensor widths.
    assert infer_tactile_layout(JOINTS + JOINTS + _tactile_names("primary")).proprio_state_dim == 12

    with pytest.raises(ValueError, match="No tactile"):
        infer_tactile_layout(JOINTS)
    shuffled = _tactile_names("primary")
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(ValueError, match="row-major"):
        infer_tactile_layout(JOINTS + shuffled)
    with pytest.raises(ValueError):
        infer_tactile_layout(JOINTS[:3] + _tactile_names("primary") + JOINTS[3:])


def test_extract_tactile_maps_strict_state_layout():
    bsz = 3
    joints = torch.randn(bsz, 6)
    left, right = torch.rand(bsz, 12, 32), torch.rand(bsz, 12, 32)
    state = torch.cat([joints, left.reshape(bsz, -1), right.reshape(bsz, -1)], dim=-1)

    proprio, maps = extract_tactile_maps_from_batch(
        {"observation.state": state}, proprio_dim=6, n_tactile_sensors=2
    )
    assert torch.equal(proprio, joints)
    assert torch.equal(maps[0], left) and torch.equal(maps[1], right)

    # Wrong layout must fail loudly instead of silently mis-slicing.
    with pytest.raises(ValueError, match="needs"):
        extract_tactile_maps_from_batch({"observation.state": state}, proprio_dim=6, n_tactile_sensors=1)


def test_extract_tactile_maps_from_stream_image_key():
    bsz = 2
    proprio, maps = extract_tactile_maps_from_batch(
        {"observation.state": torch.randn(bsz, 6), "observation.images.tactile_primary": torch.rand(bsz, 3, 96, 256)},
        tactile_keys=["observation.images.tactile_primary"],
    )
    assert proprio.shape == (bsz, 6)
    assert maps[0].shape == (bsz, 12, 32)


@pytest.mark.parametrize("as_tensor", [True, False])
def test_override_tactile_stats_only_touches_tactile_columns(as_tensor: bool):
    dim = 6 + 384
    rng = np.random.default_rng(0)
    raw = {k: rng.uniform(0.001, 0.01, size=dim) for k in ("mean", "std", "min", "max", "q01", "q99")}
    stats = {
        "observation.state": {k: torch.tensor(v) if as_tensor else v for k, v in raw.items()},
        "action": {"mean": np.zeros(6)},
    }
    out = override_tactile_stats(stats, proprio_state_dim=6)

    expected = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "q01": 0.01, "q99": 0.99}
    for key, val in expected.items():
        patched = np.asarray(out["observation.state"][key])
        np.testing.assert_allclose(patched[6:], val)
        np.testing.assert_allclose(patched[:6], raw[key][:6])
        np.testing.assert_allclose(np.asarray(stats["observation.state"][key]), raw[key])  # input untouched
    assert out["action"] is stats["action"]


def _act_tactile_config(state_dim: int = 390, **overrides):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot_policy_tactile import ACTTactileConfig

    kwargs = dict(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        chunk_size=10,
        n_action_steps=5,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=16,
        pretrained_backbone_weights=None,
        n_tactile_tokens=4,
        proprio_state_dim=6,
        n_tactile_sensors=1,
        device="cpu",
    )
    kwargs.update(overrides)
    return ACTTactileConfig(**kwargs)


def test_act_tactile_builds_once_trains_and_infers(monkeypatch: pytest.MonkeyPatch):
    from lerobot.policies.act.modeling_act import ACT
    from lerobot_policy_tactile.modeling_act_tactile import ACTTactileModel, ACTTactilePolicy

    calls = {"n": 0}
    original_init = ACT.__init__

    def counting_init(self, *args, **kwargs):
        calls["n"] += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(ACT, "__init__", counting_init)

    policy = ACTTactilePolicy(_act_tactile_config())
    assert isinstance(policy.model, ACTTactileModel)
    assert calls["n"] == 1, "the ACT backbone must be built exactly once"
    # The state token projection only sees proprioception.
    assert policy.model.encoder_robot_state_input_proj.in_features == 6

    policy.train()
    batch = {
        "observation.state": torch.randn(2, 390),
        "observation.images.top": torch.rand(2, 3, 64, 64),
        "action": torch.randn(2, 10, 6),
        "action_is_pad": torch.zeros(2, 10, dtype=torch.bool),
    }
    loss, loss_dict = policy(batch)
    assert torch.isfinite(loss) and "l1_loss" in loss_dict
    loss.backward()
    grads = [p.grad for p in policy.model.tactile_encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    policy.reset()
    action = policy.select_action(
        {"observation.state": torch.randn(1, 390), "observation.images.top": torch.rand(1, 3, 64, 64)}
    )
    assert action.shape == (1, 6)


def test_act_tactile_rejects_mismatched_state_layout():
    from lerobot_policy_tactile.modeling_act_tactile import ACTTactilePolicy

    with pytest.raises(ValueError, match="infer_tactile_layout"):
        ACTTactilePolicy(_act_tactile_config(state_dim=391))


def test_lerobot_factory_resolves_policy_and_skips_tactile_normalization():
    """End-to-end through LeRobot's own factory (processor pipelines exist in LeRobot >= 0.4)."""
    factory = pytest.importorskip("lerobot.policies.factory")
    if not hasattr(factory, "make_pre_post_processors"):
        pytest.skip("this LeRobot version normalizes inside the policy (no processor pipelines)")
    from lerobot.configs.types import NormalizationMode
    from lerobot_policy_tactile.modeling_act_tactile import ACTTactilePolicy

    assert factory.get_policy_class("act_tactile") is ACTTactilePolicy

    cfg = _act_tactile_config()
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    stats = {
        "observation.state": {"mean": torch.full((390,), 0.5), "std": torch.full((390,), 0.01)},
        "action": {"mean": torch.zeros(6), "std": torch.ones(6)},
    }
    preprocessor, _ = factory.make_pre_post_processors(cfg, dataset_stats=stats)

    state = torch.rand(1, 390)
    out = preprocessor(
        {
            "observation.state": state.clone(),
            "observation.images.top": torch.rand(1, 3, 64, 64),
            "action": torch.zeros(1, 6),
        }
    )
    normed = out["observation.state"].cpu()
    # Proprioception is normalized with the dataset stats...
    torch.testing.assert_close(normed[..., :6], (state[..., :6] - 0.5) / 0.01, rtol=1e-4, atol=1e-3)
    # ...tactile columns pass through unchanged (no division by a near-zero std).
    torch.testing.assert_close(normed[..., 6:], state[..., 6:], rtol=1e-5, atol=1e-5)
