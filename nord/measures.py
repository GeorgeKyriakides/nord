from nord.neural_nets import NeuralDescriptor

_FC_LAYER_MASS_COEFF = 1
_NON_PROC_LAYERY_MASS_FRAC = 0.1


def is_decision_layer(descriptor, layer_class):
    decision_layers = ['softmax']
    if layer_class in decision_layers:
        return True
    return False


def is_dense_layer(descriptor, layer_class):
    dense_layers = ['fc']
    if layer_class in dense_layers:
        return True
    return False


def is_io_layer(descriptor, layer_class):
    io_layers = ['input', 'output']
    if layer_class in io_layers:
        return True
    return False


def is_pooling_layer(descriptor, layer_class):
    pooling_layers = ['maxpool3x3']
    if layer_class in pooling_layers:
        return True
    return False


def get_incoming_channels(descriptor, layer_name):
    total_channels = 0

    for incoming in descriptor.incoming_connections[layer_name]:
        this_channels = 0
        layer_class = descriptor.layers[incoming][0]
        if is_pooling_layer(descriptor, layer_class):
            this_channels = get_incoming_channels(descriptor, incoming)
        else:
            layer_params = descriptor.layers[incoming][1]
            if 'out_channels' in layer_params:
                this_channels = layer_params['out_channels']
            else:
                this_channels = layer_params['num_features']

        total_channels += this_channels
    return total_channels


def get_number_of_channels(descriptor, layer_name):
    total_channels = 0
    layer_params = descriptor.layers[layer_name][1]
    if 'out_channels' in layer_params:
        total_channels = layer_params['out_channels']
    else:
        total_channels = layer_params['num_features']

    return total_channels


def get_layer_masses(d: NeuralDescriptor) -> dict:
    masses = {}

    total_mass = 0
    decision_layers = []
    io_layers = []
    for layer_name in d.layers.keys():
        layer_class = d.layers[layer_name][0]
        if is_decision_layer(d, layer_class):
            masses[layer_name] = 0
            decision_layers.append(layer_name)
            continue

        elif is_io_layer(d, layer_class):
            masses[layer_name] = 0
            io_layers.append(layer_name)
            continue

        if is_pooling_layer(d, layer_class):
            masses[layer_name] = get_incoming_channels(d, layer_name)
        else:
            masses[layer_name] = get_incoming_channels(d, layer_name) \
                * get_number_of_channels(d, layer_name)
            if is_dense_layer(d, layer_class):
                masses[layer_name] = masses[layer_name] * _FC_LAYER_MASS_COEFF

        total_mass += masses[layer_name]

    non_proc_layer_mass = max(_NON_PROC_LAYERY_MASS_FRAC * total_mass, 100)
    decision_layer_mass = non_proc_layer_mass / max(len(decision_layers), 1)

    for layer_name in decision_layers:
        masses[layer_name] = decision_layer_mass

    for layer_name in io_layers:
        masses[layer_name] = non_proc_layer_mass
    return masses
