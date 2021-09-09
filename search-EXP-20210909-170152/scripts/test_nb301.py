import numpy as np
from nord.neural_nets.nasbench301_evaluator import NASBench301Evaluator, NODE, FINAL_NODE, OPS
from nord.neural_nets import NeuralDescriptor


def create_random():
    d = NeuralDescriptor()
    for x in range(6):
        d.add_layer(NODE, None, 'NODE'+str(x))

    d.add_layer(FINAL_NODE, None, 'FINAL_NODE')

    name = 'Functional'
    for j in range(2, 6):
        if np.random.uniform() > 0.1:
            d.connect_layers('NODE'+str(j), 'FINAL_NODE')
        origins = []
        for x in range(2):
            layer = np.random.choice(OPS)
            origin = np.random.choice([x for x in range(j)])
            while origin in origins:
                origin = np.random.choice([x for x in range(j)])
            origins.append(origin)
            d.add_layer(layer, None, name+str(j)+'_'+str(x))
            d.connect_layers('NODE'+str(origin), name+str(j)+'_'+str(x))
            d.connect_layers(name+str(j)+'_'+str(x), 'NODE'+str(j))

    return d


nbe = NASBench301Evaluator()

normal = create_random()
print('Normal')
print(descriptor_to_genotype(normal))
print('Reduction')
reduction = create_random()
print(descriptor_to_genotype(reduction))


print(nbe.descriptor_evaluate(normal, reduction))
