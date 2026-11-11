"""Tests for tactile encoders, layout discovery, normalization override and act_tactile."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot_policy_tactile import (
    TactileTokenEncoder,
    extract_tactile_maps_from_batch,
    infer_tactile_layout,
    tactile_denorm_params,
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


def _lerobot_normalize(x, stats, mode, eps=1e-8):
    """Reference copy of LeRobot's NormalizerProcessorStep forward transforms."""
    if mode == "MEAN_STD":
        return (x - stats["mean"]) / (stats["std"] + eps)
    lo, hi = {"MIN_MAX": ("min", "max"), "QUANTILES": ("q01", "q99"), "QUANTILE10": ("q10", "q90")}[mode]
    d = stats[hi] - stats[lo]
    d = torch.where(d == 0, torch.full_like(d, eps), d)
    return 2 * (x - stats[lo]) / d - 1


@pytest.mark.parametrize("mode", ["MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10"])
@pytest.mark.parametrize("as_numpy", [False, True])
def test_tactile_denorm_params_invert_every_lerobot_mode(mode: str, as_numpy: bool):
    dim, proprio = 6 + 384, 6
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, dim, generator=g)
    lo = torch.rand(dim, generator=g) * 0.1
    stats = {
        "mean": torch.rand(dim, generator=g) * 0.05,
        "std": torch.rand(dim, generator=g) * 0.01,  # tiny std: the harmful case
        "min": lo, "max": lo + torch.rand(dim, generator=g),
        "q01": lo, "q99": lo + 0.5, "q10": lo, "q90": lo + 0.3,
    }
    stats["min"][10] = stats["max"][10] = 0.0  # a never-touched taxel (degenerate range)
    x[:, 10] = 0.0
    normed = _lerobot_normalize(x, stats, mode)
    passed = {k: v.numpy() for k, v in stats.items()} if as_numpy else stats
    scale, offset = tactile_denorm_params(passed, mode, start=proprio)
    torch.testing.assert_close(normed[:, proprio:] * scale + offset, x[:, proprio:], rtol=1e-4, atol=1e-4)

    with pytest.raises(ValueError):
        tactile_denorm_params(passed, "IDENTITY", start=proprio)


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


@pytest.mark.parametrize("mode", ["MEAN_STD", "MIN_MAX"])
def test_tactile_survives_lerobot_normalization_end_to_end(mode: str):
    """Through LeRobot's real normalization, whichever processor this version dispatches to.

    LeRobot 0.4.x uses ACT's processors for ACTConfig subclasses, newer versions use ours,
    0.3.x normalizes inside the policy. In every case the tactile encoder must see raw taxels.
    """
    from lerobot.configs.types import NormalizationMode
    from lerobot_policy_tactile.modeling_act_tactile import ACTTactilePolicy

    cfg = _act_tactile_config()
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode[mode],
        "ACTION": NormalizationMode.MEAN_STD,
    }
    g = torch.Generator().manual_seed(1)
    stats = {
        "observation.state": {
            "mean": torch.rand(390, generator=g) * 0.05,
            "std": torch.rand(390, generator=g) * 0.01 + 1e-4,
            "min": torch.zeros(390),
            "max": torch.rand(390, generator=g) * 0.5 + 0.5,
        },
        "action": {"mean": torch.zeros(6), "std": torch.ones(6), "min": -torch.ones(6), "max": torch.ones(6)},
    }
    policy = ACTTactilePolicy(cfg, dataset_stats=stats)
    state = torch.rand(1, 390, generator=g) * 0.5
    batch = {"observation.state": state.clone(), "observation.images.top": torch.rand(1, 3, 64, 64)}

    factory = pytest.importorskip("lerobot.policies.factory")
    if hasattr(factory, "make_pre_post_processors"):  # LeRobot >= 0.4
        from lerobot_policy_tactile.modeling_act_tactile import ACTTactilePolicy as Expected

        assert factory.get_policy_class("act_tactile") is Expected
        preprocessor, _ = factory.make_pre_post_processors(cfg, dataset_stats=stats)
        normed = preprocessor({**batch, "action": torch.zeros(1, 6)})
    else:  # LeRobot 0.3.x
        normed = policy.normalize_inputs(batch)

    normed_state = normed["observation.state"].cpu()
    assert not torch.allclose(normed_state[..., 6:], state[..., 6:]), "test expects tactile to be normalized"
    _, maps = policy.model.tactile_maps({k: v.cpu() for k, v in normed.items() if torch.is_tensor(v)})
    torch.testing.assert_close(maps[0].reshape(1, -1), state[..., 6:], rtol=1e-4, atol=1e-4)

    # The de-normalization buffers are part of the checkpoint.
    sd = policy.state_dict()
    assert {"model.tactile_denorm_scale", "model.tactile_denorm_offset", "model.tactile_denorm_ready"} <= set(sd)


def test_missing_stats_fail_loudly():
    from lerobot.configs.types import NormalizationMode
    from lerobot_policy_tactile.modeling_act_tactile import ACTTactilePolicy

    cfg = _act_tactile_config()
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.IDENTITY,
    }
    policy = ACTTactilePolicy(cfg)  # no dataset_stats, no checkpoint
    with pytest.raises(RuntimeError, match="normalization stats"):
        policy.model.tactile_maps({"observation.state": torch.rand(1, 390)})
