
"""
Created on Sat Jul 28 19:25:41 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import csv
import inspect
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

LOGS_PATH = 'Log_Files'


def singleton(class_):
    class class_w(class_):
        _instance = None

        def __new__(class_, *args, **kwargs):
            if class_w._instance is None:
                class_w._instance = super(class_w,
                                          class_).__new__(class_,
                                                          *args,
                                                          **kwargs)
                class_w._instance._sealed = False
            return class_w._instance

        def __init__(self, *args, **kwargs):
            if self._sealed:
                return
            super(class_w, self).__init__(*args, **kwargs)
            self._sealed = True
    class_w.__name__ = class_.__name__
    return class_w


def assure_reproducibility(seed=0):
    """
        Set a manual seed to pytorch and
        enable deterministic cuda execution
    """

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def plot_descriptor(descriptor):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    spacing_h = 3.0
    spacing_w = 0.5
    half_spacing = 0.25

    def my_layout(G, paths, recursions):
        nodes = G.nodes
        lengths = [-len(x) for x in paths]
        sorted_ = np.argsort(lengths)

        positions = dict()
        h = 0
        w = 0
        min_x, max_x = -spacing_w, spacing_w

        for index in sorted_:
            h = 0
            added = False
            path = paths[index]
            for node in path:
                if node not in positions:
                    positions[node] = (w, h)
                    added = True
                    h -= spacing_h
                else:
                    if h > positions[node][1]:
                        h = positions[node][1]

            if added:
                if w >= 0:
                    w += spacing_w
                w *= -1
                if w > max_x:
                    max_x = w
                if w < min_x:
                    min_x = w

        h = 0
        for node in nodes:
            if node not in positions:
                positions[node] = (w, h)
                h -= spacing_h

        f_l = descriptor.first_layer
        l_l = descriptor.last_layer
        if f_l in positions:
            positions[f_l] = (positions[f_l][0],
                              positions[f_l][1]+spacing_h)
        if l_l in positions:
            positions[l_l] = (positions[l_l][0],
                              positions[l_l][1]-spacing_h)

        recursed_nodes = []
        for path in recursions:
            last = sorted(path)[-1]
            if last not in recursed_nodes:
                positions[last] = (positions[last][0]+half_spacing,
                                   positions[last][1])
                recursed_nodes.append(last)
        return positions, min_x, max_x

    G = descriptor.to_networkx()
    plt.figure()
    plt.title(title)
    ax = plt.gca()
    in_path = descriptor.get_direct_paths()
    recs = descriptor.get_recursions()
    pos, min_x, max_x = my_layout(G, in_path, recs)

    nodes = set()
    for p in in_path:
        for node in p:
            nodes.add(node)
    for p in recs:
        for node in p:
            nodes.add(node)

    labels = {}

    for n in nodes:
        wrap_chars = 15
        name = str(descriptor.layers[n])
        labels[n] = '\n'.join(name[i:i+wrap_chars]
                              for i in range(0, len(name), wrap_chars))

    nx.draw(G, pos=pos,
            with_labels=True,
            node_shape="s",
            node_color="none",
            bbox=dict(facecolor="skyblue", edgecolor='black',
                      boxstyle='round,pad=0.2', alpha=0.5),
            labels=labels,
            font_size=8,
            nodelist=list(nodes))

    ax.set_xlim(xmin=min_x-half_spacing, xmax=max_x+half_spacing)
    plt.show()


def pdownload(url, path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(path, 'wb') as file, tqdm(
            desc=path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(loggername):
    """Get the requested loader
    """
    logs_path = LOGS_PATH
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


def get_genomes_from_logger(loggername):
    logs_path = LOGS_PATH
    loggername1 = logs_path+'/'+loggername+'.log'
    rows = []
    with open(loggername1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';', skipinitialspace=True)
        for row in csv_reader:
            row_csv_reader = csv.reader(
                row, delimiter=',', skipinitialspace=True)
            data = []
            for row_data in row_csv_reader:
                data.append(row_data)
            row_data[0] = row_data[0].replace('(', '')
            if row_data[-1][-1] == ')':
                row_data[-1] = row_data[-1][:-1]
            rows.append(row_data)
    return rows


def get_layer_out_size(in_size, kernel_size, padding=0, dilation=1, stride=1):
    # print(in_size, kernel_size, padding, dilation, stride)
    return int(np.floor((in_size+2*padding-dilation*(kernel_size-1)-1)/stride+1))


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
