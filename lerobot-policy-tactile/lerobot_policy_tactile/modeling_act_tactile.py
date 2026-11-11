"""ACT policy augmented with spatial 2D TactileTokenEncoder (`act_tactile`)."""

from __future__ import annotations

import copy
import logging
import threading

import einops
import torch
from torch import Tensor, nn

import lerobot.policies.act.modeling_act as _act_module
from lerobot.configs.types import PolicyFeature
from lerobot.policies.act.modeling_act import ACT, ACTPolicy

try:
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
except ImportError:  # LeRobot 0.3.x
    from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from .configuration_act_tactile import ACTTactileConfig
from .encoder import TactileTokenEncoder, extract_tactile_maps_from_batch
from .normalization import _mode_name, tactile_denorm_params

logger = logging.getLogger(__name__)

# ACTPolicy.__init__ builds its model via the module-level name `ACT(config)`. We swap that
# name for ACTTactileModel during construction so the network is built exactly once
# (instead of building a full ACT + ResNet and discarding it). Guarded by a lock and
# verified afterwards; falls back to explicit replacement if upstream changes.
_ACT_SWAP_LOCK = threading.Lock()


def _make_model_config(config: ACTTactileConfig) -> ACTTactileConfig:
    """Shallow copy of ``config`` with features adjusted for the ACT backbone.

    - State mode: ``observation.state`` shrinks to the proprioceptive part.
    - Key mode: tactile keys are removed so image tactile streams skip the ResNet backbone.
    Shared objects are never mutated.
    """
    if not config.use_tactile:
        return config
    features = dict(config.input_features)
    if config.tactile_in_state:
        config.validate_tactile_layout()
        state_ft = features[OBS_STATE]
        features[OBS_STATE] = PolicyFeature(type=state_ft.type, shape=(config.proprio_state_dim,))
    else:
        for key in config.tactile_keys:
            features.pop(key, None)
    model_cfg = copy.copy(config)
    model_cfg.input_features = features
    return model_cfg


class ACTTactileModel(ACT):
    """Upstream ``ACT`` plus tactile tokens appended after the 1D (latent/state) tokens."""

    def __init__(self, config: ACTTactileConfig) -> None:
        super().__init__(_make_model_config(config))
        self.tactile_config = config
        if not config.use_tactile:
            return

        self.tactile_encoder = TactileTokenEncoder(
            encoder_type=config.tactile_encoder_type,
            input_shape=config.tactile_input_shape,
            feature_dim=config.dim_model,
            n_tokens=config.n_tactile_tokens,
            dropout=config.tactile_dropout,
        )
        # Grow the 1D positional table, keeping upstream's init for the original rows.
        old = self.encoder_1d_feature_pos_embed
        extra = config.n_tactile_inputs * config.n_tactile_tokens
        new = nn.Embedding(old.num_embeddings + extra, config.dim_model)
        with torch.no_grad():
            new.weight[: old.num_embeddings] = old.weight
        self.encoder_1d_feature_pos_embed = new

        # x_raw = x_norm * scale + offset for tactile state columns (identity until stats are set).
        n_cols = config.n_tactile_state_columns
        self.register_buffer("tactile_denorm_scale", torch.ones(n_cols))
        self.register_buffer("tactile_denorm_offset", torch.zeros(n_cols))
        self.register_buffer("tactile_denorm_ready", torch.tensor(False))

    def set_tactile_normalization_stats(self, dataset_stats: dict | None) -> None:
        """Derive de-normalization buffers from the dataset stats used by the preprocessor."""
        cfg = self.tactile_config
        if not cfg.tactile_in_state:
            return
        if _mode_name(cfg.state_normalization_mode) == "IDENTITY":
            self.tactile_denorm_scale.fill_(1.0)
            self.tactile_denorm_offset.zero_()
            self.tactile_denorm_ready.fill_(True)
            return
        if not dataset_stats or OBS_STATE not in dataset_stats:
            return
        scale, offset = tactile_denorm_params(
            dataset_stats[OBS_STATE], cfg.state_normalization_mode, cfg.proprio_state_dim
        )
        if scale.shape[-1] != self.tactile_denorm_scale.shape[-1]:
            raise ValueError(
                f"observation.state stats cover {scale.shape[-1]} tactile columns, expected "
                f"{self.tactile_denorm_scale.shape[-1]}"
            )
        self.tactile_denorm_scale.copy_(scale)
        self.tactile_denorm_offset.copy_(offset)
        self.tactile_denorm_ready.fill_(True)

    def tactile_maps(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, list[Tensor]]:
        """(normalized proprio state, raw tactile maps) for an already-normalized batch."""
        cfg = self.tactile_config
        if cfg.tactile_in_state:
            if not bool(self.tactile_denorm_ready) and _mode_name(cfg.state_normalization_mode) != "IDENTITY":
                raise RuntimeError(
                    "act_tactile has no tactile normalization stats. Build the policy with dataset_stats "
                    "(lerobot-train does this via make_policy) or load a checkpoint trained that way."
                )
            state = batch[OBS_STATE]
            tactile = state[..., cfg.proprio_state_dim :] * self.tactile_denorm_scale + self.tactile_denorm_offset
            batch = {**batch, OBS_STATE: torch.cat([state[..., : cfg.proprio_state_dim], tactile], dim=-1)}
        return extract_tactile_maps_from_batch(
            batch,
            tactile_shape=cfg.tactile_input_shape,
            proprio_dim=cfg.proprio_state_dim,
            n_tactile_sensors=cfg.n_tactile_sensors,
            tactile_keys=cfg.tactile_keys or None,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        cfg = self.tactile_config
        if not cfg.use_tactile:
            return super().forward(batch)

        proprio_state, tactile_maps = self.tactile_maps(batch)
        # `self.config` is the backbone config: its image_features exclude tactile image keys.
        images = [batch[key] for key in self.config.image_features]
        batch_size = tactile_maps[0].shape[0]
        device = tactile_maps[0].device

        # 1. Latent (VAE posterior in training, zeros at inference) — mirrors upstream ACT.
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            if self.config.robot_state_feature:
                state_embed = self.vae_encoder_robot_state_input_proj(proprio_state).unsqueeze(1)
                vae_in = torch.cat([cls_embed, state_embed, action_embed], dim=1)
            else:
                vae_in = torch.cat([cls_embed, action_embed], dim=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1), False, device=device
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)
            cls_token_out = self.vae_encoder(
                vae_in.permute(1, 0, 2), pos_embed=pos_embed.permute(1, 0, 2), key_padding_mask=key_padding_mask
            )[0]
            latent_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros((batch_size, self.config.latent_dim), dtype=torch.float32, device=device)

        # 2. 1D tokens: [latent, (proprio state), (env state), *tactile tokens].
        tokens = [self.encoder_latent_input_proj(latent_sample)]
        if self.config.robot_state_feature:
            tokens.append(self.encoder_robot_state_input_proj(proprio_state))
        if self.config.env_state_feature:
            tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        for tactile_map in tactile_maps:
            tactile_tokens = self.tactile_encoder(tactile_map)  # (B, n_tokens, dim_model)
            tokens.extend(tactile_tokens[:, i] for i in range(cfg.n_tactile_tokens))
        pos = list(self.encoder_1d_feature_pos_embed.weight[: len(tokens)].unsqueeze(1))

        # 3. Camera feature-map tokens.
        for img in images:
            cam_features = self.backbone(img)["feature_map"]
            cam_pos = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            tokens.extend(einops.rearrange(cam_features, "b c h w -> (h w) b c"))
            pos.extend(einops.rearrange(cam_pos, "b c h w -> (h w) b c"))

        tokens_t = torch.stack(tokens, dim=0)
        pos_t = torch.stack(pos, dim=0)
        encoder_out = self.encoder(tokens_t, pos_embed=pos_t)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model), dtype=pos_t.dtype, device=pos_t.device
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=pos_t,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        ).transpose(0, 1)
        return self.action_head(decoder_out), (mu, log_sigma_x2)


class ACTTactilePolicy(ACTPolicy):
    """Third-party LeRobot policy (``--policy.type=act_tactile``) with spatial tactile tokens."""

    config_class = ACTTactileConfig
    name = "act_tactile"

    def __init__(self, config: ACTTactileConfig, **kwargs):
        dataset_stats = kwargs.get("dataset_stats")
        with _ACT_SWAP_LOCK:
            original = _act_module.ACT
            _act_module.ACT = ACTTactileModel
            try:
                super().__init__(config, **kwargs)
            finally:
                _act_module.ACT = original

        if not isinstance(self.model, ACTTactileModel):
            logger.warning("ACTPolicy no longer builds via `ACT(config)`; replacing the model explicitly")
            self.model = ACTTactileModel(config)
        if config.use_tactile:
            # make_policy passes the dataset stats (LeRobot >= 0.4 and 0.3.x). At inference the
            # buffers are restored from the checkpoint instead.
            self.model.set_tactile_normalization_stats(dataset_stats)
