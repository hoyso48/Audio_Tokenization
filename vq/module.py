import torch.nn as nn
from einops import rearrange
from . import activations
from .alias_free_torch import *
from torch.nn.utils import weight_norm
from torch import Tensor
import torch.nn.functional as F
import torch

class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ):
        super().__init__()
        
        # Conv1d 정의: 여기서는 padding=0으로 설정
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,          # <-- 중요!
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # 왼쪽에만 패딩을 넣어야 할 크기
        self.padding = padding
        
    def forward(self, x):
        # 입력 x의 shape은 (N, C, L)라고 가정
        # (left_padding, right_padding) 순서
        x = F.pad(x, (self.padding, 0))
        out = self.conv(x)
        return out

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNCausalConv1d(*args, **kwargs):
    # 1) 먼저 CausalConv1d 인스턴스를 만든다.
    causal_conv = CausalConv1d(*args, **kwargs)
    # 2) 내부의 self.conv를 weight_norm으로 감싸 준다.
    causal_conv.conv = weight_norm(causal_conv.conv)
    return causal_conv

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, dilations = (1, 3, 9)):
        super().__init__()
        runits = [ResidualUnit(dim // 2, dilation=d) for d in dilations]
        self.block = nn.Sequential(
            *runits,
            Activation1d(activation=activations.SnakeBeta(dim//2, alpha_logscale=True)),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            ),
        )

    def forward(self, x):
        return self.block(x)

class SSResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=kernel_size, padding=pad),
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, 4*dim, kernel_size=1),
            WNConv1d(4*dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class SSEncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, kernel_size=(3, 3)):
        super().__init__()
        runits = [SSResidualUnit(dim // 2, kernel_size=kernel_size[i]) for i in range(len(kernel_size))]
        self.block = nn.Sequential(
            *runits,
            Activation1d(activation=activations.SnakeBeta(dim//2, alpha_logscale=True)),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=stride,
                stride=stride,
                padding=stride % 2, #should be 0 #stride // 2 + stride % 2,
            )
        )

    def forward(self, x):
        return self.block(x)

class CausalResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation)
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNCausalConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class CausalEncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, dilations = (1, 3, 9)):
        super().__init__()
        runits = [CausalResidualUnit(dim // 2, dilation=d) for d in dilations]
        self.block = nn.Sequential(
            *runits,
            Activation1d(activation=activations.SnakeBeta(dim//2, alpha_logscale=True)),
            WNCausalConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride,#stride // 2 + stride % 2,
            ),
        )

    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, dilations = (1, 3, 9)):
        super().__init__()
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(input_dim, alpha_logscale=True)),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                output_padding= stride % 2,
            )
        )
        self.block.extend([ResidualUnit(output_dim, dilation=d) for d in dilations])

    def forward(self, x):
        return self.block(x)
    
class ResLSTM(nn.Module):
    def __init__(self, dimension: int,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension if not bidirectional else dimension // 2,
                            num_layers, batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, x):
        """
        Args:
            x: [B, F, T]

        Returns:
            y: [B, F, T]
        """
        x = rearrange(x, "b f t -> b t f")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = rearrange(y, "b t f -> b f t")
        return y

class ECA(nn.Module):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding="same", bias=False)

    def forward(self, inputs):
        x = inputs.mean(2)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        x = x.unsqueeze(-1)
        return inputs * x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class ScaleBiasLayer(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer('scale', torch.ones(d_model), persistent=True)
            self.register_buffer('bias', torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias

class GLU(nn.Module):
    def __init__(self, dim: int, activation: nn.Module = nn.SiLU()) -> None:
        super(GLU, self).__init__()
        self.dim = dim
        self.activation = activation

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * self.activation(gate)

class ULG(nn.Module):
    def __init__(self, dim: int, activation: nn.Module = nn.SiLU()):
        super().__init__()
        self.dim = dim
        self.activation = activation
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = torch.cat([inputs, self.activation(inputs)], dim=self.dim)
        return outputs

class GLUMlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: nn.Module = nn.SiLU()
    ) -> None:
        super(GLUMlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.glu = GLU(dim=-1, activation=activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)

        return x

class MaskedSoftmax(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # self.softmax = nn.Softmax(self.dim)

    def forward(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            # adder = (1.0 - mask.to(inputs.dtype)) * (
            #     torch.finfo(inputs.dtype).min
            # )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            # inputs += adder
            inputs = inputs.masked_fill(~mask, torch.finfo(inputs.dtype).min)
        return F.softmax(inputs, dim=self.dim)#, dtype=torch.float32)

class BlockAttention(nn.Module):
    def __init__(self, dim=256, block_size=4, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.d = dim // num_heads
        self.scale = self.d ** -0.5
        self.block_size = block_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(dropout)

    def forward(self, inputs):
        qkv = self.qkv(inputs)
        input_len = inputs.shape[1]
        if input_len % self.block_size != 0:
            qkv = F.pad(qkv, (0, 0, 0, self.block_size - input_len % self.block_size, 0, 0))
            L = qkv.shape[1]
            mask = torch.ones(L, dtype=torch.bool, device=qkv.device)
            mask[self.block_size - input_len % self.block_size:] = False
            mask = mask.view(1, 1, L//self.block_size, 1, self.block_size)
        else:
            mask = None
            L = input_len
        qkv = qkv.view(-1, L, self.num_heads, self.d * 3).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.d] * 3, dim=-1)
        # q, k, v = (B, H, L, D)
        # print(q.shape)
        if self.block_size == -1:
            block_size = L
        else:
            block_size = self.block_size
        q = q.view(-1, self.num_heads, L//block_size, block_size, self.d)
        k = k.view(-1, self.num_heads, L//block_size, block_size, self.d)
        v = v.view(-1, self.num_heads, L//block_size, block_size, self.d)

        # if mask is not None:
        #     mask = mask[:, None, None, :]

        attn = torch.matmul(q, k.permute(0, 1, 2, 4, 3)) * self.scale # (B, H, L//4, 4, 4)

        attn = MaskedSoftmax(dim=-1)(attn, mask=mask)#.to(q.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v # (B, H, L//4, 4, D)
        x = x.view(-1, self.num_heads, L, self.d)
        x = x.permute(0, 2, 1, 3).reshape(-1, L, self.dim)
        if L != input_len:
            x = x[:, :input_len, :]
        x = self.proj(x)
        # print(x.shape)
        # x = self.proj_drop(x)
        return x

class AltBlock(nn.Module):
    def __init__(self, dim=256, block_size=4, num_heads=4, expand=4, attn_dropout=0.2, mlp_dropout=0.2, drop_path=0., activation=nn.SiLU(), prenorm=False, **kwargs):
        super().__init__(**kwargs)

        self.norm1 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.self_attn = BlockAttention(dim=dim,block_size=block_size,num_heads=num_heads,dropout=attn_dropout)
        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.mlp = GLUMlp(dim, expand, dropout=mlp_dropout, activation=activation)
        self.drop2 = DropPath(drop_path)

        self.prenorm = prenorm
        self.attn_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        
    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.self_attn(x)
        x = self.drop1(x)
        x = self.attn_scale(x)
        x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        attn_out = x

        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        x = x + attn_out
        if not self.prenorm:
            x = self.norm2(x)
        x = x.transpose(1, 2)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=4, activation=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        # self.activation = activation
    def forward(self, x):
        # res = x
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x = self.activation(x)
        # x = x + res
        return x


