import os
from collections import namedtuple

import nasbench301 as nb
from nord.utils import DATA_ROOT

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


OPS = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NODE = 'NODE'
FINAL_NODE = 'FINAL_NODE'


class NASBench301Evaluator():
    """An evaluator for NASBench301.
       The descriptor should contain at least 2 'NODE' instances
       and 1 'FINAL_NODE' instance. Each operation layer (see OPS)
       should have only one incoming and one outgoing connection.
    """

    def __init__(self, version: str = '0.9') -> None:
        models_dir = DATA_ROOT+'/nb_models_'+version
        # if not os.path.exists(models_dir):
        #     os.mkdir(models_dir)

        model_paths = {
            model_name:
                models_dir+'/{}_v{}'.format(model_name, version)
            for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
        }

        # If the models are not available at the paths, automatically download
        # the models
        # Note: If you would like to provide your own model locations, comment this out
        if not all(os.path.exists(model) for model in model_paths.values()):
            nb.download_models(version=version, delete_zip=True,
                               download_dir=DATA_ROOT)

        # Load the performance surrogate model
        # NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
        # NOTE: Defaults to using the default model download path
        print("==> Loading performance surrogate model...")
        ensemble_dir_performance = model_paths['xgb']
        print(ensemble_dir_performance)
        self.performance_model = nb.load_ensemble(ensemble_dir_performance)

        # Load the runtime surrogate model
        # NOTE: Defaults to using the default model download path
        print("==> Loading runtime surrogate model...")
        ensemble_dir_runtime = model_paths['lgb_runtime']
        self.runtime_model = nb.load_ensemble(ensemble_dir_runtime)

    def descriptor_evaluate(self, regular_descriptor, reduction_descriptor):
        genotype_config = self.descriptors_to_genotype(
            regular_descriptor, reduction_descriptor)

        prediction_genotype = self.performance_model.predict(
            config=genotype_config, representation="genotype", with_noise=True)

        runtime_genotype = self.runtime_model.predict(
            config=genotype_config, representation="genotype")

        return prediction_genotype, runtime_genotype

    def descriptors_to_genotype(self, regular_descriptor, reduction_descriptor):

        normal = []
        normal_concat = []
        reduction = []
        reduction_concat = []

        def descriptor_to_genotype(descriptor):
            layers = []
            final_concat = []

            layer_names = sorted(descriptor.layers.keys())

            op_names = [
                x for x in layer_names if descriptor.layers[x][0] in OPS]
            node_names = [
                x for x in layer_names if descriptor.layers[x][0] == NODE]
            for layer in op_names:
                incoming_connections = descriptor.incoming_connections[layer]
                outgoing_connections = descriptor.connections[layer]
                for target in outgoing_connections:
                    for source in incoming_connections:
                        layers.append(
                            (descriptor.layers[layer][0], node_names.index(source)))

            concat_name = [
                x for x in layer_names if descriptor.layers[x][0] == FINAL_NODE][0]
            final_connections = descriptor.incoming_connections[concat_name]
            for source in final_connections:
                final_concat.append(node_names.index(source))

            return layers, final_concat

        normal, normal_concat = descriptor_to_genotype(regular_descriptor)
        reduction, reduction_concat = descriptor_to_genotype(
            reduction_descriptor)

        return Genotype(normal, normal_concat, reduction, reduction_concat)
