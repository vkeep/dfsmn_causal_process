# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.audio.ans.layers.layer_base import (LayerBase,
                                                           to_kaldi_matrix)
from modelscope.utils.audio.audio_utils import (expect_kaldi_matrix,
                                                expect_token_number)

COUNT = 0
class Counter:
    def __init__(self):
        self.i = 0

    def next(self, len):
        self.i += len

FLAG = Counter()
    # conv2d input shape: batch_size*chanel*time(frame_len)*features_bin
    class CausalConv2d(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                     padding_mode='zeros'):
            super().__init__(in_channels, out_channels, kernel_size,
                             stride=stride, padding=0, dilation=dilation,
                             groups=groups, bias=bias, padding_mode='zeros')
            self.__padding = (kernel_size[0] - 1) * dilation

    def forward(self, inputs):
        global PADDING
        if self.__padding == 0:
            return super(CausalConv2d)
        offset = self.__padding * inputs.size(0) * inputs.size(1) * inputs.size(3)
        pad_data_one_dim = PADDING[FLAG.i:FLAG.i + offset]
        pad_data = torch.reshape(pad_data_one_dim, (inputs.size(0), inputs.size(1), -1, inputs.size(3)))
        x = torch.cat((pad_data, inputs), -2)

        if FLAG.i == 0:
            PADDING = torch.cat((x[:, :, -self.__padding:].clone().detach().flatten(),
                                 PADDING[FLAG.i + offset:]), -1)

        PADDING[FLAG.i: FLAG.i + offset] = x[:, :, -self.__padding:, :].clone().detach().flatten()
        FLAG.next(offset)
        output = super().forward(x)
        return output

class UniDeepFsmn(LayerBase):

    def __init__(self, input_dim, output_dim, lorder=1, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        print("#####use fsmn default conv2d")
        #self.conv1 = nn.Conv2d(
        self.conv1 = CausalConv2d(
            output_dim,
            output_dim, (lorder, 1), (1, 1),
            groups=output_dim,
            bias=False)

    def forward(self, inputs):
        """

        Args:
            input: torch with shape: batch (b) x sequence(T) x feature (h)

        Returns:
            batch (b) x channel (c) x sequence(T) x feature (h)
        """

        input, buffer = inputs
        global PADDING
        PADDING = buffer.clone().detach()
        #print("the index is"+ str(FLAG.i))
        if FLAG.i >= 43776 :
            FLAG.i = 0

        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        # x: batch (b) x channel (c) x sequence(T) x feature (h)
        x_per = x.permute(0, 3, 2, 1)
        # x_per: batch (b) x feature (h) x sequence(T) x channel (c)
        #y = F.pad(x_per, [0, 0, self.lorder - 1, 0])
        conv_out = self.conv1(x_per)
        #conv_out = self.conv1(y)
        out = x_per + conv_out
        #print("#####use conv in uni deep fsmn forward")
        out1 = out.permute(0, 3, 2, 1)
        # out1: batch (b) x channel (c) x sequence(T) x feature (h)
        #print("padding len is "+ str(FLAG.i))
        return input + out1.squeeze(), PADDING

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<UniDeepFsmn> %d %d\n'\
                  % (self.output_dim, self.input_dim)
        re_str += '<LearnRateCoef> %d <HidSize> %d <LOrder> %d <LStride> %d <MaxNorm> 0\n'\
                  % (1, self.hidden_size, self.lorder, 1)

        lfiters = self.state_dict()['conv1.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        re_str += to_kaldi_matrix(x)
        proj_weights = self.state_dict()['project.weight']
        x = proj_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        return re_str

    def load_kaldi_nnet(self, instr):
        output = expect_token_number(
            instr,
            '<LearnRateCoef>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, lr = output

        output = expect_token_number(
            instr,
            '<HidSize>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, hiddensize = output
        self.hidden_size = int(hiddensize)

        output = expect_token_number(
            instr,
            '<LOrder>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, lorder = output
        self.lorder = int(lorder)

        output = expect_token_number(
            instr,
            '<LStride>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, lstride = output
        self.lstride = lstride

        output = expect_token_number(
            instr,
            '<MaxNorm>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('Fsmn format error')
        instr, mat = output
        mat1 = np.fliplr(mat.T).copy()
        print("use conv in uni fsmn load kaldi")
        self.conv1 = nn.Conv2d(
            self.output_dim,
            self.output_dim, (self.lorder, 1), (1, 1),
            groups=self.output_dim,
            bias=False)
        mat_th = torch.from_numpy(mat1).type(torch.FloatTensor)
        mat_th = mat_th.unsqueeze(1)
        mat_th = mat_th.unsqueeze(3)
        self.conv1.weight = torch.nn.Parameter(mat_th)

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, mat = output

        self.project = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.linear = nn.Linear(self.input_dim, self.hidden_size)
        self.project.weight = torch.nn.Parameter(
            torch.from_numpy(mat).type(torch.FloatTensor))

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, mat = output
        self.linear.weight = torch.nn.Parameter(
            torch.from_numpy(mat).type(torch.FloatTensor))

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, mat = output
        self.linear.bias = torch.nn.Parameter(
            torch.from_numpy(mat).type(torch.FloatTensor))
        return instr

    class SelfDeepFsmn(LayerBase):

        def __init__(self, input_dim, output_dim, lorder=1, hidden_size=None):
            super(SelfDeepFsmn, self).__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.lorder = lorder
            self.hidden_size = hidden_size


            self.linear = nn.Linear(input_dim, hidden_size)
            self.project = nn.Linear(hidden_size, output_dim, bias=False)
            print("#####use fsmn default conv2d")
            # self.conv1 = nn.Conv2d(
            self.conv1 = CausalConv2d(
                output_dim,
                output_dim, (lorder, 1), (1, 1),
                groups=output_dim,
                bias=False)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)

            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)
            self.dfsmn_net1 = UniDeepFsmn(256, 256, lorder, 256)

        def forward(self, inputs):
            """

            Args:
                input: torch with shape: batch (b) x sequence(T) x feature (h)

            Returns:
                batch (b) x channel (c) x sequence(T) x feature (h)
            """

            input, buffer = inputs
            global PADDING
            PADDING = buffer.clone().detach()
            print("the index is" + str(FLAG.i))
            # if FLAG.i >= 43776 :
            #    FLAG.i = 0

            f1 = F.relu(self.linear(input))
            p1 = self.project(f1)
            x = torch.unsqueeze(p1, 1)
            # x: batch (b) x channel (c) x sequence(T) x feature (h)
            x_per = x.permute(0, 3, 2, 1)
            # x_per: batch (b) x feature (h) x sequence(T) x channel (c)
            # y = F.pad(x_per, [0, 0, self.lorder - 1, 0])
            conv_out = self.conv1(x_per)
            # conv_out = self.conv1(y)
            out = x_per + conv_out
            print("#####use conv in uni deep fsmn forward")
            out1 = out.permute(0, 3, 2, 1)
            # out1: batch (b) x channel (c) x sequence(T) x feature (h)
            print("padding len is " + str(FLAG.i))
            return input + out1.squeeze(), PADDING
