"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
from nord.utils import get_random_value


class BlueprintGene(object):

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


class BlueprintConnectionGene(BlueprintGene):
    def __init__(self,  from_node, to_node):
        super().__init__()
        self.value = (from_node, to_node)

    def mutate(self, probability):
        if get_random_value() < probability:
            self.enabled = not self.enabled

    @staticmethod
    def __from_repr__(rpr):
        g = BlueprintConnectionGene(0, 0)

        g.innovation_number = rpr['I']
        g.value = rpr['V']
        g.enabled = rpr['E']

        return g


class BlueprintLayerGene(BlueprintGene):
    def __init__(self, layers_no, io_node=False):
        super().__init__()
        if not io_node:
            self.io = False
            self.value = get_random_value(int, 0, layers_no)
        else:
            self.value = 'IO'
            self.io = True

    def mutate(self, layers_no, probability):
        if not self.io:
            if get_random_value() < probability:
                v = get_random_value(int, 0, layers_no)
                self.value = v

    @staticmethod
    def __from_repr__(rpr):
        g = BlueprintLayerGene(1)

        g.innovation_number = rpr['I']
        g.value = rpr['V']
        g.enabled = rpr['E']

        if g.value == 'IO':
            g.io = True
        else:
            g.io = False

        return g
