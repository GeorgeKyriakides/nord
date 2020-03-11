"""
Created on 2018-08-03

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import torch.nn as nn


class Flatten(nn.Module):
    """Flattens the input, returning a 1D Tensor.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


# Grouping of layer types
convolutions = [nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]

activations = [nn.Threshold, nn.ReLU, nn.Hardtanh, nn.ReLU6, nn.Sigmoid,
               nn.Tanh, nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.ELU,
               nn.SELU, nn.Hardshrink, nn.LeakyReLU, nn.LogSigmoid,
               nn.Softplus, nn.Softshrink, nn.PReLU, nn.Softsign, nn.Softmin,
               nn.Tanhshrink, nn.RReLU, nn.GLU]

losses = [nn.L1Loss, nn.NLLLoss, nn.KLDivLoss, nn.MSELoss, nn.BCELoss,
          nn.BCEWithLogitsLoss, nn.NLLLoss2d, nn.CosineEmbeddingLoss,
          nn.HingeEmbeddingLoss, nn.MarginRankingLoss, nn.MultiLabelMarginLoss,
          nn.MultiLabelSoftMarginLoss, nn.MultiMarginLoss, nn.SmoothL1Loss,
          nn.SoftMarginLoss, nn.CrossEntropyLoss, nn.TripletMarginLoss,
          nn.PoissonNLLLoss]

pooling = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d,
           nn.MaxPool2d, nn.MaxPool3d, nn.MaxUnpool1d, nn.MaxUnpool2d,
           nn.MaxUnpool3d, nn.FractionalMaxPool2d, nn.LPPool1d, nn.LPPool2d,
           nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
           nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]

dropout = [nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout]

linear = [nn.Linear]

flatten = [Flatten]


layer_types = {'convolutions': convolutions,
               'activations': activations,
               'losses': losses,
               'pooling': pooling,
               'dropout': dropout,
               'linear': linear,
               'flatten': flatten}


def find_layer_type(layer):
    for key in layer_types.keys():
        if layer in layer_types[key]:
            return key
