import torch.nn as nn
from .darts_cell import Cell, AuxiliaryHeadCIFAR, AuxiliaryHeadImageNet, Encoder_Cell

import torch.nn.functional as F
import torch

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(3,3),padding=1):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels,eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class StackDecoder_new(nn.Module):
    def __init__(self, aux_channel, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(StackDecoder_new, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channel1 + aux_channel, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()

        # if height != 256:
        #     height = height * 2
        #     width = width * 2
        x = F.upsample(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)  # combining channels of input from encoder and upsampling input
        x = self.block(x)
        return x

class DartsMRINeuralNet(nn.Module):
    def __init__(self, C, layers, genotype, key, stem_mult=1):
        super(DartsMRINeuralNet, self).__init__()
        self._layers = layers
        self.genotype = genotype
        self.hashkey = key
        C_curr = stem_mult * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        decoder_info_list = []
        for i in range(layers):
            if i != -1:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Encoder_Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
            decoder_info_list.append([C_prev_prev, C_prev])
        print(decoder_info_list)

        self.center = ConvBlock(decoder_info_list[4][1], decoder_info_list[4][1], kernel_size=(3, 3), padding=1)

        self.up5 = StackDecoder_new(decoder_info_list[4][0], decoder_info_list[4][1], decoder_info_list[4][0],
                                    kernel_size=(3, 3))  # 32
        self.up4 = StackDecoder_new(decoder_info_list[3][0], decoder_info_list[3][1], decoder_info_list[3][0],
                                    kernel_size=(3, 3))  # 64
        self.up3 = StackDecoder_new(decoder_info_list[2][0], decoder_info_list[2][1], decoder_info_list[2][0],
                                    kernel_size=(3, 3))
        self.up2 = StackDecoder_new(decoder_info_list[1][0], decoder_info_list[1][1], decoder_info_list[1][0],
                                    kernel_size=(3, 3))
        self.up1 = StackDecoder_new(decoder_info_list[0][0], decoder_info_list[0][1], decoder_info_list[0][1],
                                    kernel_size=(3, 3))

        self.conv = nn.Conv2d(decoder_info_list[0][1], 1, kernel_size=(1, 1), bias=True)

    def forward(self, x, device=None):
        logits_aux = None
        down_list = []
        s0 = s1 = self.stem(x)
        down_list.append(s1)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, 0.0, device)
            down_list.append(s1)

        out = self.center(s1)

        out = self.up5(out, down_list[4])
        out = self.up4(out, down_list[3])
        out = self.up3(out, down_list[2])
        out = self.up2(out, down_list[1])
        out = self.up1(out, down_list[0])
        out = self.conv(out)
        return out

class DartsCifar10NeuralNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, key, stem_mult=3):
        super(DartsCifar10NeuralNet, self).__init__()
        self.hashkey = key
        self._layers = layers
        self._auxiliary = auxiliary
        self.genotype = genotype
        C_curr = stem_mult * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            # self.cells += [cell]
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x, device=None):
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob, device)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class DartsImageNetNeuralNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(DartsImageNetNeuralNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            # self.cells += [cell]
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, drop_path_prob=0.0, device=None):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, drop_path_prob, device)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux