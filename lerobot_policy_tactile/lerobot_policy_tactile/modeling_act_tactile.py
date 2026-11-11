"""ACT policy augmented with spatial 2D TactileTokenEncoder (`act_tactile`)."""

from __future__ import annotations

import copy

import einops
import torch
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTPolicy

try:
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
except ImportError:
    from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_act_tactile import ACTTactileConfig
from .encoder import TactileTokenEncoder, extract_tactile_maps_from_batch


class ACTTactileModel(ACT):
    """Extends upstream ``ACT`` by splitting flattened tactile taxels into 2D spatial tokens."""

    def __init__(self, config: ACTTactileConfig) -> None:
        # Build a shallow config copy whose robot_state_feature reflects only proprio_state_dim
        # when observation.state carries concatenated [proprio (6), tactile (384*k)].
        base_cfg = copy.deepcopy(config)
        taxels_per_pad = config.tactile_input_shape[0] * config.tactile_input_shape[1]
        if (
            config.use_tactile
            and not config.tactile_keys
            and base_cfg.robot_state_feature is not None
            and base_cfg.robot_state_feature.shape[0] > config.proprio_state_dim
        ):
            rem = base_cfg.robot_state_feature.shape[0] - config.proprio_state_dim
            if rem % taxels_per_pad == 0:
                base_cfg.input_features[OBS_STATE].shape = (config.proprio_state_dim,)

        super().__init__(base_cfg)
        self.tactile_config = config

        if config.use_tactile:
            self.tactile_encoder = TactileTokenEncoder(
                encoder_type=config.tactile_encoder_type,
                input_shape=config.tactile_input_shape,
                feature_dim=config.dim_model,
                n_tokens=config.n_tactile_tokens,
                dropout=config.tactile_dropout,
            )
            n_sensors = len(config.tactile_keys) if config.tactile_keys else config.n_tactile_sensors
            n_extra_tokens = n_sensors * config.n_tactile_tokens
            old_n_1d = self.encoder_1d_feature_pos_embed.num_embeddings
            self.encoder_1d_feature_pos_embed = nn.Embedding(old_n_1d + n_extra_tokens, config.dim_model)
            nn.init.xavier_uniform_(self.encoder_1d_feature_pos_embed.weight)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if not self.tactile_config.use_tactile:
            return super().forward(batch)

        proprio_state, tactile_maps = extract_tactile_maps_from_batch(
            batch,
            tactile_shape=self.tactile_config.tactile_input_shape,
            proprio_dim=self.tactile_config.proprio_state_dim,
            tactile_keys=self.tactile_config.tactile_keys or None,
        )
        if not tactile_maps:
            return super().forward(batch)

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif proprio_state is not None:
            batch_size = proprio_state.shape[0]
        else:
            batch_size = tactile_maps[0].shape[0]

        device = tactile_maps[0].device

        # 1. VAE latent path (training vs inference)
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            if self.config.robot_state_feature and proprio_state is not None:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(proprio_state).unsqueeze(1)
                vae_in = torch.cat([cls_embed, robot_state_embed, action_embed], dim=1)
            else:
                vae_in = torch.cat([cls_embed, action_embed], dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)
            cls_token_out = self.vae_encoder(
                vae_in.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros((batch_size, self.config.latent_dim), dtype=torch.float32, device=device)

        # 2. Assemble 1D tokens: [latent, proprio_state, (env_state), *tactile_tokens]
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        if self.config.robot_state_feature and proprio_state is not None:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(proprio_state))
        if self.config.env_state_feature and OBS_ENV_STATE in batch:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        for t_map in tactile_maps:
            t_tokens = self.tactile_encoder(t_map)  # (B, n_tactile_tokens, dim_model)
            for i in range(self.tactile_config.n_tactile_tokens):
                encoder_in_tokens.append(t_tokens[:, i])

        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight[: len(encoder_in_tokens)].unsqueeze(1)
        )

        # 3. Append 2D camera tokens if present
        if self.config.image_features and OBS_IMAGES in batch:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                encoder_in_tokens.extend(list(einops.rearrange(cam_features, "b c h w -> (h w) b c")))
                encoder_in_pos_embed.extend(list(einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")))

        stacked_tokens = torch.stack(encoder_in_tokens, dim=0)
        stacked_pos = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(stacked_tokens, pos_embed=stacked_pos)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=stacked_pos.dtype,
            device=stacked_pos.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=stacked_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        ).transpose(0, 1)

        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


class ACTTactilePolicy(ACTPolicy):
    """Third-party LeRobot policy (`--policy.type=act_tactile`) with spatial tactile tokens."""

    config_class = ACTTactileConfig
    name = "act_tactile"

    def __init__(self, config: ACTTactileConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Replace vanilla ACT model with tactile-token-enabled ACTTactileModel
        self.model = ACTTactileModel(config)

