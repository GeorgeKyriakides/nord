
"""
Created on Sat Jul 28 19:25:41 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import inspect
import logging
import sys
import time
from pathlib import Path

import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(loggername):
    """Get the requested loader
    """
    logs_path = 'Log_Files'
    Path(logs_path).mkdir(parents=True, exist_ok=True)
    loggername = logs_path+'/'+loggername
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)

    # create a file handler
    handler = logging.FileHandler(loggername+'.log')
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter(
        '%(asctime)s;%(name)s;%(levelname)s;%(message)s',
        '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger


def get_layer_out_size(in_size, kernel_size, padding=0, dilation=1, stride=1):
    # print(in_size, kernel_size, padding, dilation, stride)
    return int(np.floor((in_size+2*padding-dilation*(kernel_size-1)-1)/stride+1))


def find_optimum_pool_size(original_size, desired_size):
    kernel_size = int(np.round(original_size/desired_size))
    stride = kernel_size
    padding = 0
    while(True):
        out = get_layer_out_size(
            original_size, kernel_size, padding=padding, stride=stride)

        if out < desired_size:
            padding += 1
        elif out > desired_size:
            kernel_size += 1
        else:
            return kernel_size, stride, padding


def get_transpose_out_size(in_size, kernel_size, padding=0, dilation=1, stride=1, output_padding=0):
    return int((in_size-1)*stride - 2*padding + kernel_size + output_padding)


def extract_params(object_class, non_empty_only=False):
    """Finds the parameters of a class' constructor.

    Parameters
    ----------
    object_class : class
        The class in question.

    non_empty_only : bool (optional)
        If True, only the parameters without a default value are returned.

    Returns
    -------
    return_params : list
        A list of the requested parameters.

    """
    sig = inspect.signature(object_class.__init__)
    params = sig.parameters
    return_params = []
    for p in params:
        if non_empty_only:
            if params[p].default is inspect._empty:
                return_params.append(p)
        else:
            return_params.append(p)
    if 'self' in return_params:
        return_params.remove('self')
    return return_params


def print_all_parameters(layer_type):
    """Prints the parameters of a specific layer type.

    Parameters
    ----------
    layer_type : pytorch layer
        The layer in question.
    """
    all_params = []
    for i in layer_type:
        all_params.extend(extract_params(i))
    all_params = set(all_params)
    print(all_params)


def get_random_value(in_type=float, lower_bound=0, upper_bound=1):
    """Generate a random value of type int, float or bool.

    Parameters
    ----------
    in_type : int, float or bool
        The requested type.

    Returns
    -------
    value : int, float or bool
        The random value
    """
    if lower_bound is None:
        return 0 if upper_bound is None else upper_bound
    elif upper_bound is None:
        return 0 if lower_bound is None else lower_bound
    elif lower_bound == upper_bound:
        return lower_bound
    elif lower_bound > upper_bound:
        tmp = upper_bound
        upper_bound = lower_bound
        lower_bound = tmp

    if in_type is int:
        return np.random.randint(lower_bound, upper_bound)
    elif in_type is float:
        return np.random.uniform(lower_bound, upper_bound)
    elif in_type is bool:
        return np.random.rand() > 0.5


def generate_layer_parameters(layer, mandatory_only=True):
    """Generate random values for a given layer's class constructor parameters.

    Parameters
    ----------
    layer : class
        The requested pytorch layer class.

    mandatory_only : bool (optional)
        If True, only the parameters without a default value are returned.

    Returns
    -------
    params : list
        A list with the parameters' names.

    param_vals : list
        A list with the generated values.
    """
    from neural_nets import layers

    lt = layers.find_layer_type(layer)
    params = extract_params(layer, mandatory_only)
    param_types = []
    param_vals = []
    for param in params:
        type_ = layers.parameters[lt][param]
        param_types.append(type_)
        param_vals.append(get_random_value(type_))
    return params, param_vals


# =============================================================================
#
# =============================================================================
TOTAL_BAR_LENGTH = 65.
term_width = 40
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for _ in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for _ in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for _ in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')

    sys.stdout.write('\n')

    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
