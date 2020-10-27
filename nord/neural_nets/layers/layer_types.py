
"""
Created on Sat Jul 28 19:25:32 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import torch.nn as nn


Flatten = nn.Flatten
Identity = nn.Identity


def find_layer_type(layer):
    for key in types_dict.keys():
        if layer in types_dict[key]:
            return key


class ScaleLayer(nn.Module):

    """Scale layer.
    """

    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        return input * self.scale

    def extra_repr(self):
        return 'scale='+str(self.scale)


class SizeScaleLayer(nn.Module):

    """Size Scale layer.
    """

    def __init__(self, final_size=1):
        super().__init__()
        self.final_size = final_size

    def forward(self, input):
        return nn.functional.interpolate(input, size=(self.final_size))

    def extra_repr(self):
        return 'final_size='+str(self.final_size)


class Conv2d151(nn.Module):

    """1X5 5X1 Convolution
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, kernel_size=(1, 5),
                               stride=stride)
        self.conv7 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel_size=(5, 1),
                               stride=stride)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv7(x)
        return x

    def extra_repr(self):
        return 'in_channels='+str(self.in_channels)+', out_channels='+str(self.out_channels)


# Grouping of layer types
convolutions = [nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]

activations = [nn.Threshold, nn.ReLU, nn.Hardtanh, nn.ReLU6, nn.Sigmoid,
               nn.Tanh, nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.ELU,
               nn.SELU, nn.Hardshrink, nn.LeakyReLU, nn.LogSigmoid,
               nn.Softplus, nn.Softshrink, nn.PReLU, nn.Softsign, nn.Softmin,
               nn.Tanhshrink, nn.RReLU, nn.GLU]

# losses = [nn.L1Loss, nn.NLLLoss, nn.KLDivLoss, nn.MSELoss, nn.BCELoss,
#           nn.BCEWithLogitsLoss, nn.NLLLoss2d, nn.CosineEmbeddingLoss,
#           nn.HingeEmbeddingLoss, nn.MarginRankingLoss, nn.MultiLabelMarginLoss,
#           nn.MultiLabelSoftMarginLoss, nn.MultiMarginLoss, nn.SmoothL1Loss,
#           nn.SoftMarginLoss, nn.CrossEntropyLoss, nn.TripletMarginLoss,
#           nn.PoissonNLLLoss]

pooling = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d,
           nn.MaxPool2d, nn.MaxPool3d, nn.MaxUnpool1d, nn.MaxUnpool2d,
           nn.MaxUnpool3d, nn.FractionalMaxPool2d, nn.LPPool1d, nn.LPPool2d,
           nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
           nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]

dropout = [nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout]

linear = [nn.Linear]

flatten = [Flatten]

types_dict = {'convolutions': convolutions,
              'activations': activations,
              # 'losses': losses,
              'pooling': pooling,
              'dropout': dropout,
              'linear': linear,
              'flatten': flatten}


def get_layer_dict():
    return types_dict
