import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import BatchNormAct2d
from timm.models.layers import DropPath, trunc_normal_
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, norm_layer=BatchNormAct2d, drop_rate=0.):
        super().__init__()

        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('conv2',
                        nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        concated_features = torch.cat(x, 1)
        bottleneck_output = self.conv1(self.norm1(concated_features))  # noqa: T484
        new_features = self.conv2(self.norm2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, norm_layer=BatchNormAct2d, drop_rate=0.):
        super().__init__()

        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                norm_layer=norm_layer,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class Net(nn.Module):
    def __init__(self, pretrained=False, in_chans=3, num_classes=5, depths=[3, 3, 5, 2]):
        super().__init__()

        self.dilation_conv1 = nn.Conv2d(in_channels=in_chans,
                                        out_channels=32,
                                        kernel_size=(5, 5),
                                        padding=(4, 4),
                                        dilation=(2, 2))
        self.dense_block1 = DenseBlock(num_layers=3, num_input_features=32, bn_size=4, growth_rate=32)

        self.dilation_conv2 = nn.Conv2d(in_channels=32 + 32 * 3,
                                        out_channels=32 + 32 * 3 + 32,
                                        kernel_size=(5, 5),
                                        padding=(4, 4),
                                        dilation=(2, 2),
                                        groups=32)
        self.dense_block2 = DenseBlock(num_layers=3, num_input_features=32 + 32 * 3 + 32, bn_size=4, growth_rate=32)

        self.dilation_conv3 = nn.Conv2d(in_channels=32 + 32 * 3 + 32 + 32 * 3,
                                        out_channels=32 + 32 * 3 + 32 + 32 * 3 + 32,
                                        kernel_size=(5, 5),
                                        padding=(4, 4),
                                        dilation=(2, 2),
                                        groups=32)

        self.depths = depths
        for i in range(len(self.depths)):
            block = nn.ModuleList([Block(dim=32 + 32 * 3 + 32 + 32 * 3 + 32 * (i + 1)) for _ in range(self.depths[i])])
            norm = nn.LayerNorm(32 + 32 * 3 + 32 + 32 * 3 + 32 * (i + 1))
            conv = nn.Conv2d(in_channels=32 + 32 * 3 + 32 + 32 * 3 + 32 * (i + 1),
                             out_channels=32 + 32 * 3 + 32 + 32 * 3 + 32 * (i + 2), kernel_size=(1, 1))
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            if i != len(self.depths) - 1:
                setattr(self, f"conv{i + 1}", conv)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=32 + 32 * 3 + 32 + 32 * 3 + 32 * (len(self.depths)),
                            out_features=num_classes)

    def forward(self, x):
        x = self.dilation_conv1(x)
        x = self.dense_block1(x)
        x = self.dilation_conv2(x)
        x = self.dense_block2(x)
        x = self.dilation_conv3(x)
        for i in range(len(self.depths)):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            for blk in block:
                x = blk(x)
            B, H, W = x.shape[0], x.shape[2], x.shape[3]
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i != len(self.depths) - 1:
                conv = getattr(self, f"conv{i + 1}")
                x = conv(x)
        x = self.gap(x)
        x = self.fc(x.flatten(1))
        return x


if __name__ == '__main__':
    network = Net()
    input = torch.rand(1, 3, 224, 224)
    output = network(input)
    print(output)
