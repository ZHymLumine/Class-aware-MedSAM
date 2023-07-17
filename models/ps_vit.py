import torch
import torch.nn as nn
import math

from layers import ProgressiveSample
from .transformer_block import TransformerEncoderLayer


def load_pretrained(model, pretrained_path, num_classes, in_chans=3):
    state_dict = torch.load(pretrained_path)
    if in_chans != 3:
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    if num_classes != 1000:
        del state_dict['fc.weight']
        del state_dict['fc.bias']
    model.load_state_dict(state_dict, strict=False)

def conv3x3(in_planes,
            out_planes,
            stride=1,
            groups=1,
            dilation=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes,
            out_planes,
            stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BottleneckLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels):
        super().__init__()
        self.conv1 = conv1x1(in_channels,
                             inter_channels)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = conv3x3(inter_channels,
                             inter_channels)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = conv1x1(inter_channels,
                             out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(conv1x1(in_channels, out_channels),
                                            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PSViTLayer(nn.Module):
    def __init__(self,
                 feat_size,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 position_layer=None,
                 pred_offset=True,
                 gamma=0.1,
                 offset_bias=False):
        super().__init__()

        self.feat_size = float(feat_size)

        self.transformer_layer = TransformerEncoderLayer(dim,
                                                         num_heads,
                                                         mlp_ratio,
                                                         qkv_bias,
                                                         qk_scale,
                                                         drop,
                                                         attn_drop,
                                                         drop_path,
                                                         act_layer,
                                                         norm_layer)
        self.sampler = nn.functional.grid_sample()

        self.position_layer = position_layer
        if self.position_layer is None:
            self.position_layer = nn.Linear(2, dim)

        self.offset_layer = None
        if pred_offset:     # 迭代次数 < num_iter + 1时为True：预测offset
            self.offset_layer = nn.Linear(dim, 2, bias=offset_bias) # M_t

    def reset_offset_weight(self):
        if self.offset_layer is None:
            return
        nn.init.constant_(self.offset_layer.weight, 0)
        if self.offset_layer.bias is not None:
            nn.init.constant_(self.offset_layer.bias, 0)

    def forward(self,
                x,
                point,
                offset=None,
                pre_out=None):
        """
        dim = C
        :param x: [n, dim, h, w]
        :param point: [n, point_num, 2]     p_t
        :param offset: [n, point_num, 2]    offset    o_t
        :param pre_out: [n, point_num, dim] 上一个输出  T_{t-1}
        """
        if offset is None:
            offset = torch.zeros_like(point)

        B, C, H, W = x.shape
        B, point_num, _ = point.shape
        num_point_w = num_point_h = int(math.sqrt(point_num))
        sample_feat = self.sampler(x, point, offset) # T_t'
        #sample_feat = nn.functional.grid_sample(x, point.contiguous().view(B, num_point_w, num_point_h, _))
        #sample_feat = sample_feat.contiguous().view(B, C, num_point_w, num_point_h) # (B, C, 14, 14) [32, 384, 14, 14]
        # sample_feat = sample_feat.view(B, C, -1).transpose(1, 2)    # [32, 196, 384]
        sample_point = point + offset.detach() # p_t = p_{t-1} + o_{t-1}

        #print('sample_point: \n', sample_point)
        #print('sample_point size: ', sample_point.size(), ' self.feat_size: ', self.feat_size) # [32, 196, 384], [32, 384, 14, 14]
        pos_feat = self.position_layer(sample_point / self.feat_size) # P_t = W_t * p_t
        # print('after normalization: \n', sample_point / self.feat_size)
        # print('pos_feat: ', pos_feat.size(), ' sample_feat: ', sample_feat.size()) # [32, 196, 384]

        attn_feat = sample_feat + pos_feat  # T_t' + P_t


        if pre_out is not None:
            attn_feat = attn_feat + pre_out # final X_t = T_t' + P_t + T_{t-1}

        attn_feat = self.transformer_layer(attn_feat)   # T_t = Transformer(X_t)

        out_offset = None
        if self.offset_layer is not None:
            # print("attn_feat: ", attn_feat)
            # print('offset_layer: ', self.offset_layer, '\n, offset_layer.weight:\n', self.offset_layer.weight)
            out_offset = self.offset_layer(attn_feat)   # o_t = M_t * T_t

        # print('out_offset: ', out_offset)
        return attn_feat, out_offset, sample_point  # return T_t, o_t, p_{t+1}


class PSViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 num_point_w=14,
                 num_point_h=14,
                 in_chans=3,
                 downsample_ratio=4,
                 num_classes=1000,
                 num_iters=4,
                 depth=14,
                 embed_dim=384,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 stem_layer=None,
                 offset_gamma=0.1,
                 offset_bias=False,
                 with_cls_token=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        assert num_iters >= 1

        self.img_size = img_size
        self.feat_size = img_size // downsample_ratio   # 224/4 = 56

        self.num_point_w = num_point_w
        self.num_point_h = num_point_h

        self.register_buffer('point_coord', self._get_initial_point())

        self.pos_layer = nn.Linear(2, self.embed_dim)   # W_t

        self.stem = stem_layer
        if self.stem is None:
            self.stem = nn.Sequential(nn.Conv2d(in_chans,
                                                64,
                                                kernel_size=7,
                                                padding=3,
                                                stride=2,
                                                bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3,
                                                   stride=2,
                                                   padding=1),
                                      BottleneckLayer(64, 64, self.embed_dim),
                                      BottleneckLayer(self.embed_dim, 64, self.embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.ps_layers = nn.ModuleList()
        for i in range(num_iters):      # Progressive sampling layer
            self.ps_layers.append(PSViTLayer(feat_size=self.feat_size,
                                             dim=self.embed_dim,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop_rate,
                                             attn_drop=attn_drop_rate,
                                             drop_path=dpr[i],
                                             norm_layer=norm_layer,
                                             position_layer=self.pos_layer,
                                             pred_offset=i < num_iters - 1,
                                             gamma=offset_gamma,
                                             offset_bias=offset_bias))

        self.trans_layers = nn.ModuleList()
        trans_depth = depth - num_iters
        for i in range(trans_depth):       # Vit layers
            self.trans_layers.append(TransformerEncoderLayer(dim=self.embed_dim,
                                                             num_heads=num_heads,
                                                             mlp_ratio=mlp_ratio,
                                                             qkv_bias=qkv_bias,
                                                             qk_scale=qk_scale,
                                                             drop=drop_rate,
                                                             attn_drop=attn_drop_rate,
                                                             drop_path=dpr[i +
                                                                           num_iters],
                                                             norm_layer=norm_layer))

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)   # 分类头

        self.cls_token = None
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=.02)
        else:
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)
        # for layer in self.ps_layers:
        #     layer.reset_offset_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_initial_point(self):
        patch_size_w = self.feat_size / self.num_point_w
        patch_size_h = self.feat_size / self.num_point_h
        coord_w = torch.Tensor(
            [i * patch_size_w for i in range(self.num_point_w)])
        coord_w += patch_size_w / 2
        coord_h = torch.Tensor(
            [i * patch_size_h for i in range(self.num_point_h)])
        coord_h += patch_size_h / 2

        grid_x, grid_y = torch.meshgrid(coord_w, coord_h)
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)
        point_coord = torch.cat([grid_y, grid_x], dim=0)
        point_coord = point_coord.view(2, -1)
        point_coord = point_coord.permute(1, 0).contiguous().unsqueeze(0)

        return point_coord

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cls_token is not None:
            return {'cls_token'}
        else:
            return {}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_feature(self, x):
        batch_size = x.size(0)
        point = self.point_coord.repeat(batch_size, 1, 1) # 初始化点

        x = self.stem(x)    # extract feature map

        ps_out = None
        offset = None

        for layer in self.ps_layers:    # 迭代采样
            ps_out, offset, point = layer(x,
                                          point,
                                          offset,
                                          ps_out)

        # 最后一层输出的ps_out作为T_N, 和T_cls concat后送到Vit modules
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            trans_out = torch.cat((cls_token, ps_out), dim=1)   # concat
        else:
            trans_out = ps_out

        for layer in self.trans_layers: # vit layers
            trans_out = layer(trans_out)

        trans_out = self.norm(trans_out)

        if self.cls_token is not None:
            out_feat = trans_out[:, 0]
        else:
            trans_out = trans_out.permute(0, 2, 1)
            out_feat = self.avgpool(trans_out).view(batch_size, self.embed_dim)

        return out_feat

    def forward(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size
        # print("x shape: ", x.shape)
        x = self.forward_feature(x)

        out = self.head(x)

        return out


def _default_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


def ps_vit_b_14(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)

    stem = nn.Sequential(nn.Conv2d(kwargs.get('in_chans', 3),
                                   64,
                                   kernel_size=7,
                                   padding=3,
                                   stride=2,
                                   bias=False),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(kernel_size=3,
                                      stride=2,
                                      padding=1),
                         BottleneckLayer(64, 64, 256),
                         BottleneckLayer(256, 64, 384))

    model = PSViT(embed_dim=384,
                  num_iters=4,
                  depth=14,
                  num_heads=6,
                  mlp_ratio=3.,
                  stem_layer=stem,
                  downsample_ratio=4,
                  offset_gamma=1.0,
                  offset_bias=True,
                  with_cls_token=True,
                  **kwargs)
    model.default_cfg = _default_cfg()
    if pretrained:
        pass
    return model


def ps_vit_b_16(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)

    stem = nn.Sequential(nn.Conv2d(kwargs.get('in_chans', 3),
                                   64,
                                   kernel_size=7,
                                   padding=3,
                                   stride=2,
                                   bias=False),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(kernel_size=3,
                                      stride=2,
                                      padding=1),
                         BottleneckLayer(64, 64, 256),
                         BottleneckLayer(256, 64, 384))

    model = PSViT(embed_dim=384,
                  num_iters=4,
                  num_point_h=16,
                  num_point_w=16,
                  depth=14,
                  num_heads=6,
                  mlp_ratio=3.,
                  stem_layer=stem,
                  downsample_ratio=4,
                  offset_gamma=1.0,
                  offset_bias=True,
                  with_cls_token=True,
                  **kwargs)
    model.default_cfg = _default_cfg()
    if pretrained:
        pass
    return model


def ps_vit_b_18(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)

    stem = nn.Sequential(nn.Conv2d(kwargs.get('in_chans', 3),
                                   64,
                                   kernel_size=7,
                                   padding=3,
                                   stride=2,
                                   bias=False),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(kernel_size=3,
                                      stride=2,
                                      padding=1),
                         BottleneckLayer(64, 64, 256),
                         BottleneckLayer(256, 64, 384))

    model = PSViT(embed_dim=384,
                  num_iters=4,
                  num_point_h=18,
                  num_point_w=18,
                  depth=14,
                  num_heads=6,
                  mlp_ratio=3.,
                  stem_layer=stem,
                  downsample_ratio=4,
                  offset_gamma=1.0,
                  offset_bias=True,
                  with_cls_token=True,
                  **kwargs)
    model.default_cfg = _default_cfg()
    if pretrained:
        pass
    return model



def ps_vit_ti_14(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 192 ** -0.5)

    stem = nn.Sequential(nn.Conv2d(kwargs.get('in_chans', 3),
                                   64,
                                   kernel_size=7,
                                   padding=3,
                                   stride=2,
                                   bias=False),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True),
                         nn.MaxPool2d(kernel_size=3,
                                      stride=2,
                                      padding=1),
                         BottleneckLayer(64, 64, 192),
                         BottleneckLayer(192, 64, 192))

    model = PSViT(embed_dim=192,
                  num_iters=4,
                  depth=12,
                  num_heads=3,
                  mlp_ratio=3.,
                  stem_layer=stem,
                  downsample_ratio=4,
                  offset_gamma=1.0,
                  offset_bias=True,
                  with_cls_token=True,
                  **kwargs)
    model.default_cfg = _default_cfg()
    if pretrained:
        pass
    return model
