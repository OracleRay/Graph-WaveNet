import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))  # 将输入与邻接矩阵相乘
        return x.contiguous()  # 确保结果张量在内存中是连续存储的


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()  # 图卷积层
        c_in = (order * support_len + 1) * c_in  # 计算多阶卷积的输入通道数
        self.mlp = linear(c_in, c_out)  # 线性层
        self.dropout = dropout
        self.order = order  # 图卷积的阶数

    def forward(self, x, support):
        out = [x]  # 第 0 阶的卷积结果
        # 扩散卷积
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)  # 将所有卷积结果按通道维度拼接
        h = self.mlp(h)  # 将拼接后的特征传入线性层 mlp，得到转换后的节点特征。
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool  # 是否开启图卷积
        self.addaptadj = addaptadj

        # ModuleList: 专门用于保存一组神经网络层的容器
        self.filter_convs = nn.ModuleList()  # 过滤器卷积层
        self.gate_convs = nn.ModuleList()  # 门控卷积层
        self.residual_convs = nn.ModuleList()  # 残差卷积层
        self.skip_convs = nn.ModuleList()  # 跳跃连接的卷积层
        self.bn = nn.ModuleList()  # 批归一化层
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports  # 存放邻接矩阵的列表

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:  # 判断是否使用自适应矩阵
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)  # 奇异值分解(m:左奇异矩阵,p:奇异值向量,n:右奇异矩阵的转置)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field  # 模型的感受野

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            # 在时间维度上填充输入，保证序列长度达到感受野的要求
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)  # 一个 1x1 卷积层，将输入特征变换到 residual_channels 大小
        skip = 0  # 用于存储每层的跳跃连接（skip connections）结果

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None  # 存放新邻接矩阵（自适应矩阵）的列表
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # 计算自适应矩阵
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x  # 保存输入，实现残差连接

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)  # 生成一个候选信息
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)  # 生成“门控信号”,控制候选信息在当前层中保留的比例
            x = filter * gate  # 扩展卷积的输出

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]  # 确保尺寸对齐
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)  # 带自适应矩阵的图卷积
                else:
                    x = self.gconv[i](x, self.supports)  # 普通图卷积
            else:
                x = self.residual_convs[i](x)  # 常规卷积

            x = x + residual[:, :, :, -x.size(3):]  # 残差连接

            x = self.bn[i](x)  # 批归一化，保持特征稳定

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
