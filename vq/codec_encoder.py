import torch
from torch import nn
import numpy as np
from .module import WNConv1d, WNCausalConv1d, EncoderBlock, SSEncoderBlock, ResLSTM, AltBlock, Downsample
from .alias_free_torch import *
from . import activations

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class BigCodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                use_rnn=True,
                rnn_bidirectional=False,
                causal=False,
                rnn_num_layers=2,
                up_ratios=(2, 2, 2, 5, 5),
                dilations=(1, 3, 9),
                out_channels=1024):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        if causal:
            assert not rnn_bidirectional

        # Create first convolution
        d_model = ngf
        if causal:
            self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=6)]
        else:
            self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, dilations=dilations, causal=causal)]
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(d_model,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        # Create last convolution
        if causal:
            self.block += [
                Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
                WNCausalConv1d(d_model, out_channels, kernel_size=3, padding=2),
            ]
        else:
            self.block += [
                Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
                WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
            ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

class SSBigCodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                use_rnn=True,
                rnn_bidirectional=False,
                causal=False,
                rnn_num_layers=2,
                up_ratios=(2, 2, 2, 5, 5),
                kernel_sizes=(3, 3),
                out_channels=1024):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        if causal:
            assert not rnn_bidirectional

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=1)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [SSEncoderBlock(d_model, stride=stride, kernel_sizes=kernel_sizes, causal=causal)]
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(d_model,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        # Create last convolution
        self.block += [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        self.reset_parameters()

    def forward(self, x):
        # print(x.shape)
        # for i, block in enumerate(self.block):
        #     x = block(x)
        #     print(x.shape)
        # out = x
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

class BlockCodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                use_rnn=True,
                rnn_bidirectional=False,
                rnn_num_layers=2,
                up_ratios=(2, 2, 2, 5, 5),
                dilations=(1, 3, 9),
                out_channels=1024,
                blocks=1,
                dropout=0.0,
                drop_rate=0.0,
                expand=4,
        ):
        super().__init__()
        stages = len(up_ratios)
        # ngf = 2*ngf
        self.dim = ngf
        channels = [ngf * (2**i) for i in range(stages)]
        self.hidden_dim = out_channels
        # print(channels)
        
        # Initial projection
        # self.stem_conv = nn.Conv1d(in_channels, channels[0], 
        #                           kernel_size=kernel_size, stride=strides[0], 
        #                           bias=True, padding=kernel_size//2)
        # self.stem_bn = nn.BatchNorm1d(channels[0], momentum=0.05)
        # self.init_conv = nn.Conv1d(1, ngf, kernel_size=1)
        self.stem_conv = nn.Conv1d(1, 
                            channels[0], 
                            kernel_size=up_ratios[0], 
                            stride=up_ratios[0], 
                            bias=True)
        self.stem_bn = nn.LayerNorm(channels[0], eps=1e-6)
        # self.stem_conv = nn.Conv1d(in_channels, channels[0], 
        #                     kernel_size=kernel_size, stride=strides[0], 
        #                     bias=True, padding=(kernel_size-strides[0],0))

        # Encoder blocks
        self.stages = nn.ModuleList()
        for i in range(stages):
            stage = nn.ModuleList()
            
            # First block with stride
            if i > 0:
                # stage.append(EncoderBlock(
                #     in_channels=channels[i-1],
                #     out_channels=channels[i],
                #     kernel_size=strides[i],
                #     stride=strides[i],
                #     dropout=dropout,
                #     drop_rate=drop_rate,
                #     expand=expand,
                #     activation=activation,
                #     causal=causal
                # ))
                stage.append(Downsample(channels[i-1], 
                channels[i], 
                kernel_size=up_ratios[i], 
                stride=up_ratios[i], 
                activation=nn.SiLU(), 
                ))
            
            # Additional blocks
            if i < stages - 1:
                for _ in range(blocks):
                    block_size = 5 #up_ratios[i+1]
                    assert block_size <= np.prod(up_ratios[i+1:])
                    stage.append(AltBlock(
                        dim=channels[i],
                        block_size=block_size,#np.prod(up_ratios[i+1:]),
                        num_heads=4,
                        expand=4,
                        attn_dropout=dropout,
                        mlp_dropout=drop_rate,
                        drop_path=drop_rate,
                        activation=nn.SiLU(),
                        prenorm=False,
                    ))

            self.stages.append(stage)
        self.stages.append(nn.ModuleList([nn.Conv1d(channels[-1], out_channels, kernel_size=1)]))
        
    def forward(self, x):
        # x = self.init_conv(x)
        x = self.stem_conv(x)
        x = x.transpose(1, 2)
        x = self.stem_bn(x)
        x = x.transpose(1, 2)
        # print(x.shape)
        
        # Process each stage
        # print(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
                # print(x.shape)
        return x

    def inference(self, x):
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        # def _apply_weight_norm(m):
        #     if isinstance(m, nn.Conv1d):
        #         torch.nn.utils.weight_norm(m)

        # self.apply(_apply_weight_norm)
        pass

    def reset_parameters(self):
        self.apply(init_weights)