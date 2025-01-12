import torch
from torch import nn

from ultralytics.yolo.utils.plotting import feature_visualization
from .FastSAM.fastsam import FastSAM
from torch.nn import functional as F
from typing import Dict, List
from utils.misc import initialize_weights
import matplotlib.pyplot as plt
import math

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.SA(x)
        return A


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class Mix(nn.Module):
    def __init__(self, m=-0.8):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1-mix_factor.expand_as(fea2))
        return out

class SAM_CD(nn.Module):
    def __init__(
            self,
            num_embed=8,
            model_name: str = '/path/to/fastsam_model/FastSAM-x.pt',
            device: str = 'cuda',
            conf: float = 0.4,
            iou: float = 0.9,
            imgsz: int = 256,
            retina_masks: bool = True,
            done_warmup: bool = True,
    ):
        super(SAM_CD, self).__init__()
        self.model = FastSAM(model_name)
        self.device = device
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_feats = None


        self.Adapter32 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter16 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter8 = nn.Sequential(nn.Conv2d(320, 80, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(80), nn.ReLU())
        self.Adapter4 = nn.Sequential(nn.Conv2d(160, 40, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(40), nn.ReLU())


        # self.Adapter32 = nn.Sequential(nn.Conv2d(512, 160, kernel_size=1, stride=1, padding=0, bias=False),
        #                                nn.BatchNorm2d(160), nn.ReLU())
        # self.Adapter16 = nn.Sequential(nn.Conv2d(256, 160, kernel_size=1, stride=1, padding=0, bias=False),
        #                                nn.BatchNorm2d(160), nn.ReLU())
        # self.Adapter8 = nn.Sequential(nn.Conv2d(128, 80, kernel_size=1, stride=1, padding=0, bias=False),
        #                               nn.BatchNorm2d(80), nn.ReLU())
        # self.Adapter4 = nn.Sequential(nn.Conv2d(64, 40, kernel_size=1, stride=1, padding=0, bias=False),
        #                               nn.BatchNorm2d(40), nn.ReLU())

        self.Dec2 = _DecoderBlock(160, 160, 80)
        self.Dec1 = _DecoderBlock(80, 80, 40)
        self.Dec0 = _DecoderBlock(40, 40, 64)
        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1)

        self.SA = Space_Attention(16, 16, 4)
        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1)
        self.resCD = self._make_layer(ResBlock, 128, 128, 6, stride=1)
        self.headC = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU())
        self.segmenterC = nn.Conv2d(16, 1, kernel_size=1)



        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0 = conv_block_nested(40, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(80, filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(160, filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(160, filters[3], filters[3])
        self.Up3_0 = up(filters[3])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2, filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(filters[0] * 4, 1, kernel_size=1)

        # self.fusion = nn.Conv2d(2, 1, kernel_size=1)

        self.mix = Mix()

        for param in self.model.model.parameters():
            param.requires_grad = False


    def run_encoder(self, image):
        self.image = image
        feats = self.model(
            self.image,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )
        return feats

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):

        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        featsB = self.run_encoder(x2)

        featA_s4 = self.Adapter4(featsA[3].clone())
        featA_s8 = self.Adapter8(featsA[0].clone())
        featA_s16 = self.Adapter16(featsA[1].clone())
        featA_s32 = self.Adapter32(featsA[2].clone())

        decA_2 = self.Dec2(featA_s32, featA_s16)
        decA_1 = self.Dec1(decA_2, featA_s8)
        decA_0 = self.Dec0(decA_1, featA_s4)
        outA = self.segmenter(decA_0)

        # visualization(outA, name='A')

        featB_s4 = self.Adapter4(featsB[3].clone())
        featB_s8 = self.Adapter8(featsB[0].clone())
        featB_s16 = self.Adapter16(featsB[1].clone())
        featB_s32 = self.Adapter32(featsB[2].clone())

        decB_2 = self.Dec2(featB_s32, featB_s16)
        decB_1 = self.Dec1(decB_2, featB_s8)
        decB_0 = self.Dec0(decB_1, featB_s4)
        outB = self.segmenter(decB_0)
        # visualization(outB,name='B')

        x0_0A = self.conv0_0(featA_s4)
        x1_0A = self.conv1_0(featA_s8)
        x2_0A = self.conv2_0(featA_s16)
        x3_0A = self.conv3_0(featA_s32)

        x0_0B = self.conv0_0(featB_s4)
        x1_0B = self.conv1_0(featB_s8)
        x2_0B = self.conv2_0(featB_s16)
        x3_0B = self.conv3_0(featB_s32)

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        A = self.SA(torch.cat([outA, outB], dim=1))
        featC = torch.cat([decA_0, decB_0], 1)
        featC = self.resCD(featC)
        featC = self.headC(featC) * A
        outC = self.segmenterC(featC)

        # fusion = torch.cat([out, outC], dim=1)
        # outF = self.fusion(fusion)
        outF = self.mix(out, outC)
        return F.interpolate(out, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outC, input_shape, mode="bilinear", align_corners=True), \
               F.interpolate(outF, input_shape, mode="bilinear", align_corners=True), \
               F.interpolate(outA, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outB, input_shape, mode="bilinear", align_corners=True)



def visualization(x, n=8, name='',save_dir='/home/ziyuan/wzb/image2/'):
        """
        Visualize feature maps of a given model module during inference.

        Args:
            x (torch.Tensor): Features to be visualized.
            module_type (str): Module type.
            stage (int): Module stage within the model.
            n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
            save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
        """

        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir + name + ".png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 4), 4, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            for i in range(n):
                # 反转亮度
                feature_map = blocks[i].squeeze()  # 去掉通道维度，得到 2D 图像

                # 归一化到 [0, 1] 范围
                feature_map = feature_map - feature_map.min()
                feature_map = feature_map / feature_map.max()

                # 反转颜色
                feature_map = 1 - feature_map  # 反转亮度：原来暗的地方变亮，亮的地方变暗

                ax[i].imshow(feature_map)  # cmap='gray' 保持灰度显示
                ax[i].axis('off')

            # for i in range(n):
            #     ax[i].imshow(blocks[i].squeeze(),cmap='hot')  # cmap='gray'
            #     ax[i].axis('off')
            plt.savefig(f, dpi=150, bbox_inches='tight')
            plt.close()
