"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
from nord.utils import get_random_value


class Gene(object):

    def __init__(self):
        self.innovation_number = None
        self.enabled = True
        self.value = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value and self.innovation_number == other.innovation_number
        return False

    def __hash__(self):
        return hash(str(self.value))

    @staticmethod
    def __from_repr__(rpr):
        raise NotImplementedError

    def __repr__(self):
        return str({'I': self.innovation_number,
                    'V': self.value,
                    'E': self.enabled})

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
                if self.bound_types[i] is bool:
                    self.value[i] = not self.value[i]
                else:

                    v = get_random_value(self.bound_types[i],
                                         self.bounds[0][i],
                                         self.bounds[1][i])

                    self.value[i] = v

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
