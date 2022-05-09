from nord.neural_nets import NATSBench_Evaluator, NeuralDescriptor

ne = NATSBench_Evaluator()
descriptor = NeuralDescriptor()
descriptor.add_layer("IN", None, "IN")
descriptor.add_layer_sequential("nor_conv_3x3", None, "1")
descriptor.add_layer_sequential("nor_conv_1x1", None, "2")
descriptor.add_layer_sequential("avg_pool_3x3", None, "3")
# descriptor.add_layer_sequential("avg_pool_3x3", None, "4") # does not work with 5 nodes
descriptor.add_layer_sequential("OUT", None, "OUT")

ne._descriptor_to_nasnet(descriptor)

descriptor.add_layer("skip_connect", None, "4")
descriptor.connect_layers("1", "4")
descriptor.connect_layers("4", "OUT")

ne._descriptor_to_nasnet(descriptor)

descriptor.add_layer("skip_connect", None, "5")
descriptor.connect_layers("IN", "5")
descriptor.connect_layers("5", "3")

ne._descriptor_to_nasnet(descriptor)

descriptor.add_layer("nor_conv_3x3", None, "6")
descriptor.connect_layers("IN", "6")
descriptor.connect_layers("6", "OUT")

ne._descriptor_to_nasnet(descriptor)

ne.descriptor_evaluate(descriptor)

ne.api.query_index_by_arch(ne._descriptor_to_nasnet(descriptor))


for k in descriptor.connections.keys():
    outs = descriptor.connections[k]
    for out in outs:
        if out not in descriptor.incoming_connections:
            descriptor.incoming_connections[out] = []
        incs = descriptor.incoming_connections[out]
        incs.append(k)
        descriptor.incoming_connections[out] = incs
