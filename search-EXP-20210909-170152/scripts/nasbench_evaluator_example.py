"""
Example usage of BenchmarkEvaluator,
which evaluates the given architecture in NASBench-101

"""
from nord.neural_nets import BenchmarkEvaluator, NeuralDescriptor

# Instantiate the evaluator
evaluator = BenchmarkEvaluator()
# Instantiate a descriptor
d = NeuralDescriptor()

# See the available layers (ops)
# for NASBench-101
layers = evaluator.get_available_ops()
print(layers)

# Add NASBench-101 Layers connected
# sequentially
d.add_layer('input', None, 'in')
d.add_layer_sequential(layers[0], None, 'layer_1')
d.add_layer_sequential(layers[2], None, 'layer_2')
d.add_layer_sequential('output', None, 'out')


# Add Connections
d.connect_layers('layer_1', 'out')

# Get the validation accuracy and training time
val_acc, train_time = evaluator.descriptor_evaluate(
    d, acc='validation_accuracy')

# Get the test accuracy and training time
test_acc, train_time = evaluator.descriptor_evaluate(d, acc='test_accuracy')

print(d)
print('Train time:', train_time)
print('Validation accuracy:', val_acc)
print('Test accuracy:', test_acc)
