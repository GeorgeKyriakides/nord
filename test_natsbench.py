from nord.neural_nets import NATSBench_Evaluator, NeuralDescriptor

ne = NATSBench_Evaluator()
descriptor = NeuralDescriptor()
descriptor.add_layer('IN', None, 'IN')
descriptor.add_layer_sequential('avg_pool_3x3', None, '1')
descriptor.add_layer_sequential('nor_conv_1x1', None, '2')
descriptor.add_layer_sequential('skip_connect', None, '3')
descriptor.add_layer_sequential('OUT', None, 'OUT')

descriptor.add_layer('avg_pool_3x3', None, '4')
descriptor.connect_layers('1', '4')
descriptor.connect_layers('4', 'OUT')

descriptor.add_layer('nor_conv_3x3', None, '5')
descriptor.connect_layers('IN', '5')
descriptor.connect_layers('5', '3')


descriptor.add_layer('nor_conv_3x3', None, '6')
descriptor.connect_layers('IN', '6')
descriptor.connect_layers('6', 'OUT')

ne.descriptor_evaluate(descriptor)
