"""
Created on 2018-08-05

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

DEFAULT_SUFFIX = 'Layer'


class NeuralDescriptor(object):
    """A descriptor class of a network's layer types and connections.
    """

    def __init__(self):
        self.layers = {}
        self.connections = {}
        self.current_layer = 0
        self.first_layer = ''
        self.last_layer = ''

    def add_layer(self, layer_class, parameters, name=None):
        """Adds a layer type with specific parameters to the network.

        Parameters
        ----------

        layer_class : class
            The class of the added layer.

        parameters : list
            The parameters of the added layer.

        name : str (optional)
            The name of the added layer.

        """

        if name is None:
            name = self.__get_layer_name()

        if self.current_layer == 0:
            self.first_layer = name

        self.last_layer = name
        self.layers[name] = [layer_class, parameters]
        self.connections[name] = []
        self.current_layer += 1

    def add_layer_sequential(self, layer_class, parameters, name=None):
        """Adds a layer type with specific parameters to the network,
            while connecting it to the previously added layer.

        Parameters
        ----------

        layer_class : class
            The class of the added layer.

        parameters : list
            The parameters of the added layer.

        name : str (optional)
            The name of the added layer.

        """
        previous_layer = self.last_layer
        self.add_layer(layer_class, parameters, name)
        if not previous_layer == '':
            self.connect_layers(previous_layer, self.last_layer)

    def connect_layers(self, from_layer, to_layer):
        """Connects two layers existing in the graph, by using their names.

        Parameters
        ----------
        from_layer : str
            The name of the origin.

        to_layer : str
            The name of the destination.
        """
        self.connections[from_layer].append(to_layer)

    def __get_layer_name(self):
        """Generates the next auto-generated layer name.
        """
        return str(self.current_layer)+'_'+DEFAULT_SUFFIX
