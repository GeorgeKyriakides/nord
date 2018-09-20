"""
Created on 2018-08-03

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
# Type descriptions for each layer's parameters


convolutions = {'bias': bool,
                'dilation': int,
                'groups': int,
                'in_channels': int,
                'kernel_size': int,
                'out_channels': int,
                'output_padding': int,
                'padding': int,
                'stride': int}

activations = {'min_value': float,
               'inplace': bool,
               'num_parameters': int,
               'init': float,
               'value': float,
               'dim': int,
               'lambd': float,
               'negative_slope': float,
               'threshold': float,
               'max_value': float,
               'beta': float,
               'alpha': float,
               'lower': float,
               'max_val': float,
               'upper': float,
               'min_val': float}

losses = {'p': int,
          'log_input': bool,
          'full': bool,
          'size_average': bool,
          'reduce': bool,
          'ignore_index': bool,
          'margin': float,
          'eps': float,
          'swap': bool}

pooling = {'count_include_pad': bool,
           'ceil_mode': bool,
           'output_size': int,
           'output_ratio': [0.0, 0.1],
           'return_indices': bool,
           'kernel_size': int,
           'norm_type': float,
           'stride': int,
           'dilation': int,
           'padding': int}

dropout = {'inplace': bool,
           'p': float}

linear = {'in_features': int,
          'out_features': int,
          'bias': bool}


layer_parameters = {'convolutions': convolutions,
                    'activations': activations,
                    'losses': losses,
                    'pooling': pooling,
                    'dropout': dropout,
                    'linear': linear}
