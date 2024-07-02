from typing import Optional, Tuple

import torch
import torch.nn as nn

from diffusers.utils import is_torch_version
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
)
from .unet_2d_blocks import get_down_block, get_up_block


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class EncoderSkip(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        additional_in_channels: int = 0,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.additional_in_channels = additional_in_channels

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if additional_in_channels>0:
            self.add_conv_in = nn.Conv2d(
                additional_in_channels,
                block_out_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.down_block_skip_convs = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        skip_conv = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        skip_conv = zero_module(skip_conv)
        self.down_block_skip_convs.append(skip_conv)
        
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)
            
            for _ in range(layers_per_block):
                skip_conv = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                skip_conv = zero_module(skip_conv)
                self.down_block_skip_convs.append(skip_conv)

            if not is_final_block:
                skip_conv = nn.Conv2d(block_out_channels[i], block_out_channels[i+1], kernel_size=1)
                skip_conv = zero_module(skip_conv)
                self.down_block_skip_convs.append(skip_conv)

        # mid
        mid_block_channel = block_out_channels[-1]
        skip_conv = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        skip_conv = zero_module(skip_conv)
        self.mid_block_skip_conv = skip_conv
        
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conv_out_channels, 3, padding=1
        )

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        r"""The forward method of the `Encoder` class."""
        # sample : (B, C, H, W)
        if hasattr(self, "add_conv_in"):
            res_sample = self.add_conv_in(sample[:, self.in_channels:, :, :])
        else:
            res_sample = 0
        sample = res_sample + self.conv_in(sample[:, :self.in_channels, :, :])
        
        down_block_res_samples = (sample,)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample, res_samples = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                    down_block_res_samples += res_samples
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
                mid_block_res_samples = sample.clone()
            else:
                for down_block in self.down_blocks:
                    sample, res_samples = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample
                    )
                    down_block_res_samples += res_samples
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample
                )
                mid_block_res_samples = sample.clone()

        else:
            # down
            for down_block in self.down_blocks:
                sample, res_samples = down_block(sample)
                down_block_res_samples += res_samples

            # middle
            sample = self.mid_block(sample)
            mid_block_res_samples = sample.clone()

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        skip_down_block_res_samples = ()
        for down_block_res_sample, skip_conv in zip(down_block_res_samples, self.down_block_skip_convs):
            skip_down_block_res_samples = skip_down_block_res_samples + (skip_conv(down_block_res_sample),)
        down_block_res_samples = skip_down_block_res_samples
        mid_block_res_samples = self.mid_block_skip_conv(mid_block_res_samples)

        return sample, down_block_res_samples, mid_block_res_samples



class DecoderSkip(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.FloatTensor,
        down_block_res_samples: Optional[Tuple[torch.FloatTensor, ...]] = None,
        mid_block_res_samples: Optional[torch.FloatTensor] = None,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                if mid_block_res_samples is not None:
                    sample = sample + mid_block_res_samples
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    res_samples = None
                    if down_block_res_samples is not None:
                        res_samples = down_block_res_samples[-len(up_block.resnets) :]
                        down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        res_samples,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                if mid_block_res_samples is not None:
                    sample = sample + mid_block_res_samples
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    res_samples = None
                    if down_block_res_samples is not None:
                        res_samples = down_block_res_samples[-len(up_block.resnets) :]
                        down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, res_samples, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            if mid_block_res_samples is not None:
                sample = sample + mid_block_res_samples
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                res_samples = None
                if down_block_res_samples is not None:
                    res_samples = down_block_res_samples[-len(up_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
                sample = up_block(sample, res_samples, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

