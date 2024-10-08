import torch.nn as nn
import torch
import torch.nn.functional as F

# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
from torch.autograd import Variable
import math
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm
from torch.distributions import Bernoulli
from .windowattention import WindowAttention,BasicLayer
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)


        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:

            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
    
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv3x3(planes, planes)    
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size


    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNetzuiyuanshi(nn.Module):

    def __init__(self, args, block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        self.args = args
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # transformer
        self.conv_3 = nn.Conv2d(320, 640, kernel_size=4, padding=1, stride=2)
        self.encoder_layer_3 = nn.TransformerEncoderLayer(d_model=640, nhead=args.head, dropout=args.dp1)
        self.decoder_layer_3 = nn.TransformerDecoderLayer(d_model=640, nhead=args.head, dropout=args.dp1)
        # self.windowatt1=WindowAttention(640,(5,5),8)
        # self.windowatt2=WindowAttention(640,(5,5),8)
        self.windowatt=BasicLayer(640, 2, 8, 5)
    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.args, self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)        
        x = self.layer2(x)        
        x = self.layer3(x) 
        # =========================== CIFE ============================
        # mem_3 = self.conv_3(x)
        # mem_3 = self.windowatt(mem_3.reshape(mem_3.shape[0], mem_3.shape[1], -1).transpose(-1,-2),W=5,H=5)[0]
        #
        # x = self.layer4(x)
        # tgt_3 = x.reshape(x.shape[0], x.shape[1], -1).transpose(-1, -2)
        # tgt_3= self.windowatt(tgt_3,W=5,H=5)[0]
        # x4 = x + 0.1*self.decoder_layer_3(tgt_3, mem_3).transpose(-1, -2).reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        return x

class ResNet(nn.Module):

    def __init__(self, args, block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        self.args = args
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # transformer
        # self.conv_3 = nn.Conv2d(64, 640, kernel_size=5, padding=0, stride=4)
        self.conv_1 = nn.Conv2d(64, 640, kernel_size=3, padding=1, stride=2)
        self.conv_2 = nn.Conv2d(160, 640, kernel_size=3, padding=1, stride=2)

        self.conv_3 = nn.Conv2d(320, 640, kernel_size=4, padding=1, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.encoder_layer_3 = nn.TransformerEncoderLayer(d_model=640, nhead=args.head, dropout=args.dp1)
        self.decoder_layer_3 = nn.TransformerDecoderLayer(d_model=640, nhead=args.head, dropout=args.dp1)
        # self.windowatt1=WindowAttention(640,(5,5),8)
        # self.windowatt2=WindowAttention(640,(5,5),8)
        self.windowatt=BasicLayer(640, 2, 8, 5)
    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.args, self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)#80，64，21， 21
        mem_1 = self.conv_1(x)
        mem_1 = self.adaptive_pool(mem_1)
        mem_1 = self.windowatt(mem_1.reshape(mem_1.shape[0], mem_1.shape[1], -1).transpose(-1,-2),W=5,H=5)[0]
        x = self.layer2(x)#80，160，21，21
        mem_2 = self.conv_2(x)
        mem_2 = self.adaptive_pool(mem_2)
        mem_2 = self.windowatt(mem_2.reshape(mem_2.shape[0], mem_2.shape[1], -1).transpose(-1,-2),W=5,H=5)[0]

        x = self.layer3(x)#80，320，10，10
        # =========================== CIFE ============================

        mem_3 = self.conv_3(x)
        mem_3 = self.adaptive_pool(mem_3)
        mem_3 = self.windowatt(mem_3.reshape(mem_3.shape[0], mem_3.shape[1], -1).transpose(-1,-2),W=5,H=5)[0]
        x = self.layer4(x)
        mem_4 = x.reshape(x.shape[0], x.shape[1], -1).transpose(-1, -2)
        mem_4= self.windowatt(mem_4,W=5,H=5)[0]


        mem_1_3=mem_1+mem_3
        mem_2_4=mem_2+mem_4

        x4 = x + 0.1*self.decoder_layer_3(mem_1_3,mem_2_4).transpose(-1, -2).reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        return x4


class ResNet18(nn.Module):
    maml = False  # Default

    def __init__(self, args, block=SimpleBlock, list_of_num_layers=[2, 2, 2, 2], list_of_out_dims=[64, 128, 256, 512], flatten=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet18, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0) and i != 3
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            # self.final_feat_dim = indim

        self.feat_dim = [512, 14, 14]
        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        # out = out.view(out.size(0), -1)
        return out


if __name__=='__main__':
    model=ResNet()
    input = torch.FloatTensor(5, 3, 80, 80)
    out = model(input)
