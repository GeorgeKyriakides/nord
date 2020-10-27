# nord
Deep neural architecture research framework


NORD is a deep neural architecture research framework that aims to provide tools in order to 
make the implementation and comparison of neural architecture design methods fair and straightforward.

## Main concepts

- Design Algorithms: Designing neural network specifications

- Descriptors: Making the programmatical generation of nodes and connections more straigth-forward.

- Evaluators: Evaluating the quality of descriptors' networks.

- Environments: Managing the distributed execution of Evaluators.

## Main requirements

- PyTorch and Torchvision (https://pytorch.org/)

- NetworkX

-Tensorflow 1.15 for the NASBench-101 benchmark dataset


## Examples

- descriptor_example.py Example, usage of NeuralDescriptor and NeuralNet classes
- local_evaluator_example.py, Example usage of LocalEvaluator, which evaluates the given architecture on various datasets.
- nasbench_evaluator_example.py, Example usage of BenchmarkEvaluator,which evaluates the given architecture in NASBench-101.
- reproducible_genetic_algorithm_example_cifar10.py, Example of a reproducible simple genetic algorithm based on DeepNEAT.
- custom_dataset_example.py, Example of evaluating on a custom dataset

