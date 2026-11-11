"""Unit tests for spatial tactile token encoders and batch slicing."""

from __future__ import annotations

import pytest
import torch

from lerobot_policy_tactile import (
    TactileAttentionCNN,
    TactileCNN,
    TactileTokenEncoder,
    extract_tactile_maps_from_batch,
)


@pytest.mark.parametrize("encoder_type", ["cnn", "attention"])
@pytest.mark.parametrize("n_tokens", [1, 4])
def test_tactile_token_encoder_forward_and_gradients(encoder_type: str, n_tokens: int):
    enc = TactileTokenEncoder(
        encoder_type=encoder_type,
        input_shape=(12, 32),
        feature_dim=256,
        n_tokens=n_tokens,
    )
    x = torch.rand(4, 12, 32, requires_grad=True)
    tokens = enc(x)
    assert tokens.shape == (4, n_tokens, 256)

    loss = tokens.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_extract_tactile_maps_from_flattened_state():
    # Simulate SOTactileFollower batch: 6 joint dims + 2 sensors * (12 * 32) = 774 dims
    bsz = 3
    joints = torch.randn(bsz, 6)
    pad_left = torch.rand(bsz, 12, 32)
    pad_right = torch.rand(bsz, 12, 32)
    state = torch.cat([joints, pad_left.reshape(bsz, -1), pad_right.reshape(bsz, -1)], dim=-1)

    proprio, maps = extract_tactile_maps_from_batch(
        {"observation.state": state},
        tactile_shape=(12, 32),
        proprio_dim=6,
    )
    assert proprio is not None and proprio.shape == (bsz, 6)
    assert len(maps) == 2
    assert torch.allclose(maps[0], pad_left)
    assert torch.allclose(maps[1], pad_right)


def test_extract_tactile_maps_from_stream_image_key():
    bsz = 2
    img = torch.rand(bsz, 3, 96, 256)
    proprio, maps = extract_tactile_maps_from_batch(
        {
            "observation.state": torch.randn(bsz, 6),
            "observation.images.tactile_primary": img,
        },
        tactile_shape=(12, 32),
        proprio_dim=6,
        tactile_keys=["observation.images.tactile_primary"],
    )
    assert proprio is not None and proprio.shape == (bsz, 6)
    assert len(maps) == 1
    assert maps[0].shape == (bsz, 12, 32)


def test_act_tactile_policy_forward_backward_and_select_action():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot_policy_tactile.configuration_act_tactile import ACTTactileConfig
    from lerobot_policy_tactile.modeling_act_tactile import ACTTactilePolicy

    # 6 joint dims + 384 tactile taxels = 390 dims in observation.state
    cfg = ACTTactileConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(390,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
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
        use_tactile=True,
        tactile_encoder_type="attention",
        n_tactile_tokens=4,
        proprio_state_dim=6,
        device="cpu",
    )

    policy = ACTTactilePolicy(cfg)
    policy.train()

    batch = {
        "observation.state": torch.randn(2, 390),
        "observation.images.top": torch.rand(2, 3, 64, 64),
        "action": torch.randn(2, 10, 6),
        "action_is_pad": torch.zeros(2, 10, dtype=torch.bool),
    }
    loss, loss_dict = policy(batch)
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
    loss.backward()

    # Verify gradients actually flow through TactileTokenEncoder!
    tactile_grads = [
        p.grad for p in policy.model.tactile_encoder.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(tactile_grads) > 0
    assert all(torch.isfinite(g).all() for g in tactile_grads)

    # Verify inference select_action works on single step
    policy.reset()
    action = policy.select_action(
        {
            "observation.state": torch.randn(1, 390),
            "observation.images.top": torch.rand(1, 3, 64, 64),
        }
    )
    assert action.shape == (1, 6)

