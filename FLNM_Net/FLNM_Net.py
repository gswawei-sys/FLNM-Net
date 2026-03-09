import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchvision.transforms import RandomCrop
import math

ds = (1, 2, 2)  # Spatial downsampling_stride  # noqa
dp = (1, 0, 0)  # Spatial downsampling_padding # noqa

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, groups=1, act=nn.Tanh):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.norm = nn.InstanceNorm3d(out_channel)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# Frequency-Adaptive Modulator
class FAG(nn.Module):
    def __init__(self, channels, device, seq_len=160, freq_range=(0.5, 3.0)):
        super().__init__()
        self.seq_len = seq_len
        self.freq_min, self.freq_max = freq_range
        self.freq = nn.Parameter(torch.ones(channels, device=device))          # Frequency-related parameters
        self.bandwidth = nn.Parameter(torch.zeros(channels, device=device))
        self.amplitude = nn.Parameter(torch.zeros(channels, 3, device=device)) # Amplitude parameter
        t = torch.arange(seq_len, device=device) / seq_len
        self.register_buffer('t', t)

    def forward(self, x):
        # Calculate the center frequency
        center_freq = self.freq_min + (self.freq_max - self.freq_min) * torch.sigmoid(self.freq)
        # Calculate the bandwidth
        bandwidth = 0.1 + 0.9 * F.softplus(self.bandwidth)
        # Generate the main frequency and two sidebands
        freqs = torch.stack([center_freq, center_freq - bandwidth / 2, center_freq + bandwidth / 2], dim=1)
        # Generate three frequency components
        waves = torch.sin(2 * math.pi * freqs.view(-1, 3, 1) * self.t)
        # Amplitude weighting
        amplitude = F.softmax(self.amplitude, dim=1)
        oscillation = (waves * amplitude.view(-1, 3, 1)).sum(dim=1)
        x = x * oscillation.view(1, -1, self.seq_len, 1, 1)

        return x

# Luminance-Noise Aware MASK
class LNAM(nn.Module):
    def __init__(self, channels, adaptive_gamma=True, gamma=0.9):
        super().__init__()
        self.eps = 1e-6
        self.w = 0.7
        self.w1 = nn.Parameter(torch.tensor(self.w))
        self.w2 = nn.Parameter(torch.tensor(1.0 - 0.3 * self.w))

        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma probability must be between 0 and 1, got {gamma}")
        self.register_buffer('gamma_param', torch.tensor(gamma))
        self.adaptive_gamma = adaptive_gamma
        if self.adaptive_gamma:
            self.gamma_P = nn.Parameter(torch.tensor(gamma))
        else:
            self.gamma_P = gamma

        self.temporal_filter = nn.Conv3d(
            in_channels=channels, out_channels=channels,
            kernel_size=(5, 1, 1),
            stride=1, padding=(5 // 2, 0, 0),
            groups=channels, bias=False
        )

        with torch.no_grad():
            w = torch.ones(channels, 1, 5, 1, 1)
            w /= 5
            self.temporal_filter.weight.copy_(w)

    def Filtering_Feature_Refinement(self, x): # Feature_Refinement
        decomposed_x = self.temporal_filter(x)
        residual = x - decomposed_x
        refined = torch.sigmoid(self.w1) * residual + torch.sigmoid(self.w2) * (decomposed_x + x) / 2
        mean = refined.mean(dim=2, keepdim=True)
        std = refined.std(dim=2, keepdim=True).add(self.eps)
        x = (refined - mean) / std
        return x

    def forward(self, x):
        if self.adaptive_gamma:
            self.gamma = torch.sigmoid(self.gamma_P)
        else:
            self.gamma = self.gamma_P

        if self.gamma == 0 or not self.training:
            return x
        else:
            B, C, T, H, W = x.shape

            # Lighting estimation
            illum = x.mean(dim=1, keepdim=True).mean(dim=[3, 4], keepdim=True)
            light_weight = 1 - illum  # [B,1,T,1,1]
            light_weight = light_weight.mean(dim=2, keepdim=True)
            light_weight = light_weight.expand(-1, C, -1, -1, -1)

            # Inter-frame noise estimation
            diff = x[:, :, 1:] - x[:, :, :-1]
            diff_score = diff.pow(2).mean(dim=[2, 3, 4], keepdim=True)
            noise_weight = (diff_score - diff_score.min()) / (diff_score.max() - diff_score.min() + 1e-6)

            # Weight modulation
            mask_weight = light_weight * noise_weight
            mask_weight = mask_weight.max() * self.gamma
            mask = torch.empty(x.shape[:2] + (1, 1, 1), dtype=x.dtype, device=x.device).uniform_(0, 1)
            mask = (mask < mask_weight) / (mask_weight + 1e-6)

            # # # Feature Refinement
            x = self.Filtering_Feature_Refinement(x)
            x = x * mask
            return x

# Deep Average Temporal Attention
class DATA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, use_att=True, reduction=4, act=nn.Tanh):
        super().__init__()
        self.main_path = ConvBlock(in_channel, out_channel, kernel_size, stride, padding, act=act)

        self.use_att = use_att
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=5, padding=2, groups=in_channel, bias=False),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        bottleneck_channel = max(out_channel // reduction, 1)
        self.channel_mapper = nn.Sequential(
            nn.Linear(in_channel, bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channel, out_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main_path.conv(x)
        if self.use_att:
            spatial_pooled = x.mean(dim=[3, 4])
            temporal_feats = self.temporal_attention(spatial_pooled).squeeze(-1)
            attn = self.channel_mapper(temporal_feats).view(x.shape[0], -1, 1, 1, 1)
            out = out * attn
            out = self.main_path.norm(out)
            out = self.main_path.act(out)
        return out

class FLNM_Net(nn.Module):
    def __init__(self, in_channel, gamma=0.9, use_att=True, use_FAG=True, adaptive_gamma=True, device=torch.device("cpu")):
        super(FLNM_Net, self).__init__()

        self.use_FAG = use_FAG
        self.adaptive_gamma = adaptive_gamma
        self.FFR = LNAM(3)

        if self.use_FAG:
            self.Feature_res = ConvBlock(10, 10, 3, ds, dp)
            self.FAG = FAG(10, device)

        self.Feature_extraction = nn.Sequential(
            ConvBlock(in_channel, 8, 3, 1, 1),      # 1, 8 , 160, 72, 72
            ConvBlock(8, 10, 3, ds, dp),            # 1, 10, 160, 35, 35
        )

        self.Conv_layer_1 = nn.Sequential(
            ConvBlock(10, 10, 3, 1, dp),            # 1, 10, 160, 32, 32
            ConvBlock(10, 10, 3, ds, dp),           # 1, 10, 160, 16, 16
        )

        self.Conv_layer_2 = ConvBlock(10, 16, 3, 1, dp)  # 1, 16, 160, 14, 14

        self.Conv_layer_3 = nn.Sequential(
            DATA(16, 16, 3, 1, dp, use_att=use_att), # 1, 16, 160, 10, 10
            ConvBlock(16, 16, 3, 1, dp),             # 1, 16, 160, 10, 10
            ConvBlock(16, 16, 3, 1, dp),             # 1, 16, 160, 8 , 8
        )

        self.Conv_layer_4 = nn.Sequential(
            DATA(16, 16, 3, 1, dp, use_att=use_att),# 1, 16, 160, 6 , 6
            ConvBlock(16, 10, 3, 1, dp),            # 1, 10, 160, 4 , 4
            ConvBlock(10, 8, 3, 1, dp),             # 1, 8 , 160, 2 , 2
        )

        self.Get_rppg = nn.Conv3d(8, 1, (3, 2, 2), stride=1, padding=dp, bias=False)  # 1, 1, 160, 1, 1

        self.LNAM1 = LNAM(10, adaptive_gamma=adaptive_gamma, gamma=gamma)
        self.LNAM2 = LNAM(16, adaptive_gamma=adaptive_gamma, gamma=gamma)
        self.LNAM3 = LNAM(16, adaptive_gamma=adaptive_gamma, gamma=gamma)

    def forward(self, x):
        N, C, D, W, H = x.shape                         # 1, 8 , 160, 72, 72
        if self.use_FAG:
            x = self.FFR.Filtering_Feature_Refinement(x)

        x_FE = self.Feature_extraction(x)               # 1, 10, 160, 35, 35
        x1 = self.Conv_layer_1(x_FE)                    # 1, 10, 160, 16, 16
        x1 = self.LNAM1(x1)                             # 1, 10, 160, 16, 16
        if self.use_FAG:                                # Frequency Adaptive Gate (FAG)
            x_res = self.Feature_res(x_FE)              # 1, 10, 160, 17, 17
            transform = RandomCrop(size=(16, 16))       # 1, 10, 160, 16, 16
            x1 = x1 + self.FAG(transform(x_res))        # 1, 10, 160, 16, 16

        x2 = self.Conv_layer_2(x1)                      # 1, 16, 160, 14, 14
        x2 = self.LNAM2(x2)                             # 1, 16, 160, 14, 14
        x3 = self.Conv_layer_3(x2)                      # 1, 16, 160, 8 , 8
        x3 = self.LNAM3(x3)                             # 1, 16, 160, 8 , 8

        x4 = self.Conv_layer_4(x3)                      # 1, 8 , 160, 2 , 2
        rPPG = self.Get_rppg(x4)                        # 1, 1 , 160, 1 , 1
        rPPG = rPPG.squeeze(-1).squeeze(-1).squeeze(1)  # 1, 1 , 160
        return rPPG

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    import numpy as np

    input_x = np.load(
        r"G:\PreprocessingDataset\RD\UBFC-PHYS_SizeW72_SizeH72_ClipLength160_DataTypeRD_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse\DiffNormalized\s5_T2_input0.npy")
    input_x = torch.from_numpy(input_x)
    input_x = input_x.unsqueeze(0)
    input_x = input_x.permute(0, 4, 1, 2, 3)
    input_x = input_x.to(device).float()
    # input_x = torch.rand([4, 3, 160, 72, 72]).to(device)
    print("Input dtype:", input_x.dtype)
    print(input_x.device)

    # 使用自适应gamma的模型
    model = FLNM_Net(in_channel=3, gamma=0.9, use_att=True, use_FAG=True, adaptive_gamma=True, device=device)
    model = model.to(device)
    output_tensor = model(input_x)
    print('Input size:', input_x.size())
    print("Output shape:", output_tensor.shape)

    # 计算 FLOPs 和参数量
    from thop import profile
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        flops, params = profile(model, inputs=(input_x,))

    model_size_MB = (params * 4) / (1024 ** 2)
    print(f"Total FLOPs: {flops / 1e9:.2f} G")
    print(f"Model Parameters: {params / 1e6:.2f} M")
    print(f"Model Size: {model_size_MB :.5f} MB")

    # 打印可学习的gamma参数
    for name, param in model.named_parameters():
        if 'gamma' in name:
            print(f"Adaptive Gamma - Name: {name}, Value: {torch.sigmoid(param).item():.4f}, Raw: {param.item():.4f}")
        else:
            print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")