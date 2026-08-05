"""Conditional encoders for the two released SRDiff variants."""

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = inputs.shape
        sequences = inputs.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, frames, channels
        )
        outputs, _ = self.attn(sequences, sequences, sequences)
        return outputs.reshape(batch, height, width, frames, channels).permute(
            0, 4, 3, 1, 2
        ).contiguous()


class BaseConditionAdapter(nn.Module):
    """Base spatiotemporal adapter: [B,T,3,H,W] -> [B,T,4,H/4,W/4]."""

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        out_channels: int = 4,
        num_residual_blocks: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[Residual3DBlock(mid_channels) for _ in range(num_residual_blocks)]
        )
        self.temp_attn = TemporalAttention(mid_channels, num_heads=num_heads)
        self.res_block_post_attn = Residual3DBlock(mid_channels)
        self.conv3d_2 = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        outputs = self.relu(self.bn1(self.conv3d_1(outputs)))
        outputs = self.res_blocks(outputs)
        outputs = self.temp_attn(outputs)
        outputs = self.res_block_post_attn(outputs)
        outputs = self.relu(self.bn2(self.conv3d_2(outputs)))
        return outputs.permute(0, 2, 1, 3, 4).contiguous()


class SingleModalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 64,
        num_residual_blocks: int = 2,
    ):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[Residual3DBlock(mid_channels) for _ in range(num_residual_blocks)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.res_blocks(self.relu(self.bn1(self.conv3d_1(inputs))))


def _mask_logits(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return inputs + (1.0 - mask.float()) * -1e30


class Conv1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv1d(inputs.transpose(1, 2)).transpose(1, 2)


class CQAttention(nn.Module):
    """Context-query attention used by the modality attention module."""

    def __init__(self, dim: int, drop_rate: float = 0.0):
        super().__init__()
        self.w4C = nn.Parameter(torch.empty(dim, 1))
        self.w4Q = nn.Parameter(torch.empty(dim, 1))
        self.w4mlu = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.w4C)
        nn.init.xavier_uniform_(self.w4Q)
        nn.init.xavier_uniform_(self.w4mlu)
        self.dropout = nn.Dropout(drop_rate)
        self.cqa_linear = Conv1D(4 * dim, dim)

    def forward(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        context_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        score = self._trilinear_attention(context, query)
        score_query = torch.softmax(_mask_logits(score, query_mask.unsqueeze(1)), dim=2)
        score_context = torch.softmax(
            _mask_logits(score, context_mask.unsqueeze(2)), dim=1
        ).transpose(1, 2)
        context_to_query = torch.matmul(score_query, query)
        query_to_context = torch.matmul(
            torch.matmul(score_query, score_context), context
        )
        fused = torch.cat(
            [
                context,
                context_to_query,
                context * context_to_query,
                context * query_to_context,
            ],
            dim=2,
        )
        return self.cqa_linear(fused)

    def _trilinear_attention(
        self, context: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        context = self.dropout(context)
        query = self.dropout(query)
        context_length = context.shape[1]
        query_length = query.shape[1]
        context_score = torch.matmul(context, self.w4C).expand(-1, -1, query_length)
        query_score = (
            torch.matmul(query, self.w4Q).transpose(1, 2).expand(-1, context_length, -1)
        )
        interaction = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        return context_score + query_score + interaction


class MTAI(nn.Module):
    """Modality attention followed by temporal self-attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        patch_size: int = 2,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=None,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=channels,
        )
        self.cq_attn = CQAttention(channels, drop_rate=drop_rate)
        self.unpatch_proj = nn.Linear(channels, channels * patch_size * patch_size)
        self.mam_norm = nn.LayerNorm(channels)
        self.tam = TemporalAttention(channels, num_heads=num_heads)
        self.tam_norm = nn.BatchNorm3d(channels)

    def _unpatchify(
        self, tokens: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        batch, _, channels = tokens.shape
        patch = self.patch_size
        patch_height, patch_width = height // patch, width // patch
        tokens = self.unpatch_proj(tokens).reshape(
            batch, patch_height, patch_width, channels, patch, patch
        )
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        return tokens.reshape(batch, channels, height, width)

    def forward(
        self, primary: torch.Tensor, auxiliary: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, frames, height, width = primary.shape
        primary_2d = primary.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        auxiliary_2d = auxiliary.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        primary_tokens = self.patch_embed(primary_2d)
        auxiliary_tokens = self.patch_embed(auxiliary_2d)
        token_mask = torch.ones(
            primary_tokens.shape[:2],
            device=primary.device,
        )
        fused = self.cq_attn(
            primary_tokens,
            auxiliary_tokens,
            token_mask,
            token_mask,
        )
        fused = self.mam_norm(fused)
        fused = self._unpatchify(fused, height, width)
        fused = fused.reshape(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        ).contiguous()
        fused = primary + fused
        return self.tam_norm(fused + self.tam(fused))


class CMCAConditionAdapter(nn.Module):
    """Paper CMCA: two modality encoders, two MTAIs, and post-fusion encoding."""

    def __init__(
        self,
        mid_channels: int = 64,
        out_channels: int = 4,
        num_residual_blocks: int = 2,
        num_mtai_blocks: int = 2,
        num_heads: int = 4,
        patch_size: int = 2,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.ir_encoder = SingleModalEncoder(
            in_channels=2,
            mid_channels=mid_channels,
            num_residual_blocks=num_residual_blocks,
        )
        self.lght_encoder = SingleModalEncoder(
            in_channels=1,
            mid_channels=mid_channels,
            num_residual_blocks=num_residual_blocks,
        )
        self.mtai_blocks = nn.ModuleList(
            [
                MTAI(
                    channels=mid_channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    drop_rate=drop_rate,
                )
                for _ in range(num_mtai_blocks)
            ]
        )
        self.post_conv = nn.Sequential(
            nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.post_res_blocks = nn.Sequential(
            *[Residual3DBlock(mid_channels) for _ in range(num_residual_blocks)]
        )
        self.final_conv = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        infrared = inputs[:, :, :2].permute(0, 2, 1, 3, 4).contiguous()
        lightning = inputs[:, :, 2:].permute(0, 2, 1, 3, 4).contiguous()
        infrared = self.ir_encoder(infrared)
        lightning = self.lght_encoder(lightning)
        fused = infrared
        for mtai in self.mtai_blocks:
            fused = mtai(fused, lightning)
        fused = self.post_res_blocks(self.post_conv(fused))
        fused = self.final_conv(fused)
        return fused.permute(0, 2, 1, 3, 4).contiguous()
