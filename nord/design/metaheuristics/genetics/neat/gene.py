"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
from nord.utils import get_random_value


DISABLE_PROB = 0.75
PERTURBE_PROB = 0.9


class Gene(object):

    def __init__(self):
        self.innovation_number = None
        self.enabled = True
        self.value = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value  # and self.enabled == other.enabled
        return False

    def __hash__(self):
        return hash(str(self.value))

    # def __str__(self):
    #     en = 'E' if self.enabled else 'D'
    #     return ('[I-'+str(self.innovation_number) +
    #             '|V-'+str(self.value) +
    #             '|'+en+']')

    @staticmethod
    def __from_repr__(rpr):
        raise NotImplementedError

    def __repr__(self):
        return str({'I': self.innovation_number,
                    'V': self.value,
                    'E': self.enabled})

    def crossover(self, other):
        new_gene = None
        # Select to copy from this or from the other parent
        if get_random_value(bool):
            new_gene = self.copy()

        else:
            new_gene = other.copy()

        # Probability that it remains disabled
        if not new_gene.enabled:
            if get_random_value() > DISABLE_PROB:
                new_gene.enabled = True

        return new_gene

    def is_homologous(self, other):
        if self.innovation_number == other.innovation_number:
            return True
        return False

    def mutate(self, probability):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class ConnectionGene(Gene):
    def __init__(self,  from_node, to_node):
        super().__init__()
        self.value = (from_node, to_node)

    def mutate(self, probability):
        if get_random_value() < probability:
            self.enabled = not self.enabled

    @staticmethod
    def __from_repr__(rpr):
        g = ConnectionGene(0, 0)

        g.innovation_number = rpr['I']
        g.value = rpr['V']
        g.enabled = rpr['E']

        return g

    def copy(self):
        g = ConnectionGene(*self.value)
        g.innovation_number = self.innovation_number
        g.enabled = self.enabled
        return g


class LayerGene(Gene):
    def __init__(self, bound_types, bounds, io_node=False):
        super().__init__()
        if not io_node:
            self.io = False
            layer_parameters = []
            self.bounds = bounds
            self.bound_types = bound_types
            # Iterate for each parameter
            for i in range(len(bound_types)):
                v = get_random_value(
                    bound_types[i], bounds[0][i], bounds[1][i])
                layer_parameters.append(v)
            self.value = layer_parameters
        else:
            self.value = 'IO'
            self.io = True
            self.bound_types = []
            self.bounds = []

    def mutate(self, probability):
        if not self.io:
            if get_random_value() < probability:
                i = get_random_value(int, 0, len(self.value))
                if get_random_value() > PERTURBE_PROB or self.bound_types[i] is bool:
                    v = get_random_value(self.bound_types[i],
                                         self.bounds[0][i],
                                         self.bounds[1][i])
                    self.value[i] = v
                else:

                    v = get_random_value(self.bound_types[i],
                                         self.bounds[0][i],
                                         self.bounds[1][i])
                    new_v = self.value[i]+v

                    new_v = max(self.bounds[0][i], new_v)
                    new_v = min(self.bounds[1][i], new_v)
                    self.value[i] = new_v

    @staticmethod
    def __from_repr__(rpr):
        g = LayerGene([int], [[0], [1]])

        g.innovation_number = rpr['I']
        g.value = rpr['V']
        g.enabled = rpr['E']

        if g.value == 'IO':
            g.io = True
        else:
            g.io = False

        return g

    def copy(self):
        g = LayerGene(self.bound_types, self.bounds, self.io)
        g.innovation_number = self.innovation_number
        g.enabled = self.enabled
        g.value = self.value
        return g


class HyperparametersGene(object):
    def __init__(self, bound_types, bounds):
        super().__init__()

        parameters = []
        self.bounds = bounds
        self.bound_types = bound_types
        # Iterate for each parameter
        for i in range(len(bound_types)):
            v = get_random_value(
                bound_types[i], bounds[0][i], bounds[1][i])
            parameters.append(v)
        self.value = parameters

    def mutate(self, probability):
        if get_random_value() < probability:
            i = get_random_value(int, 0, len(self.value))
            if get_random_value() > PERTURBE_PROB or self.bound_types[i] is bool:
                v = get_random_value(self.bound_types[i],
                                     self.bounds[0][i],
                                     self.bounds[1][i])
                self.value[i] = v
            else:
                v = get_random_value(self.bound_types[i],
                                     self.bounds[0][i]/10,
                                     self.bounds[1][i]/10)
                new_v = self.value[i]+v

                new_v = max(self.bounds[0][i], new_v)
                new_v = min(self.bounds[1][i], new_v)
                self.value[i] = new_v
