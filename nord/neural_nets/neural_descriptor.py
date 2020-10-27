
"""
Created on Mon Aug  6 18:54:35 2018

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
        self.incoming_connections = {}
        self.current_layer = 0
        self.first_layer = ''
        self.last_layer = ''

    def add_layer(self, layer_class: type, parameters: dict, name: str = None):
        """Adds a layer type with specific parameters to the network.

        Parameters
        ----------

        layer_class : class
            The class of the added layer.

        parameters : dict
            The dictionary of parameters of the added layer.

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
        self.incoming_connections[name] = []
        self.current_layer += 1

    def add_layer_sequential(self, layer_class: type, parameters: dict,
                             name: str = None):
        """Adds a layer type with specific parameters to the network,
            while connecting it to the previously added layer.

        Parameters
        ----------

        layer_class : class
            The class of the added layer.

        parameters : dict
            The dictionary of parameters of the added layer.

        name : str (optional)
            The name of the added layer.

        """
        previous_layer = self.last_layer
        self.add_layer(layer_class, parameters, name)
        if not previous_layer == '':
            self.connect_layers(previous_layer, self.last_layer)

    def connect_layers(self, from_layer: str, to_layer: str):
        """Connects two layers existing in the graph, by using their names.

        Parameters
        ----------
        from_layer : str
            The name of the origin.

        to_layer : str
            The name of the destination.
        """
        self.connections[from_layer].append(to_layer)
        self.incoming_connections[to_layer].append(from_layer)

    def disconnect_layers(self, from_layer: str, to_layer: str):
        """Disconnects two layers existing in the graph, by using their names.

        Parameters
        ----------
        from_layer : str
            The name of the origin.

        to_layer : str
            The name of the destination.
        """
        while to_layer in self.connections[from_layer]:
            self.connections[from_layer].remove(to_layer)
            if from_layer in self.incoming_connections[to_layer]:
                self.incoming_connections[to_layer].remove(from_layer)

    def __get_layer_name(self):
        """Generates the next auto-generated layer name.
        """
        return str(self.current_layer)+'_'+DEFAULT_SUFFIX

    def add_suffix(self, suffix: str):
        """Adds suffix to all layers and connections.
        """
        self.first_layer = self.first_layer+suffix
        self.last_layer = self.last_layer+suffix
        layers = list(self.layers.keys())
        for key in layers:
            self.layers[key+suffix] = self.layers.pop(key)

        connections = list(self.connections.keys())
        for connection in connections:
            this_conn = self.connections[connection]
            new_conn = []
            for conn in this_conn:
                new_conn.append(conn+suffix)
            self.connections[connection+suffix] = new_conn
            self.connections.pop(connection)

        incoming_connections = list(self.incoming_connections.keys())
        for connection in incoming_connections:
            this_conn = self.incoming_connections[connection]
            new_conn = []
            for conn in this_conn:
                new_conn.append(conn+suffix)
            self.incoming_connections[connection+suffix] = new_conn
            self.incoming_connections.pop(connection)

    def __repr__(self):
        return str({'Layers': repr(self.layers), 'Connections': repr(self.connections),
                    'First_Layer': self.first_layer, 'Last_Layer': self.last_layer})

    def to_networkx(self):
        """
        Return a networkx graph representation of
        this descriptor

        Returns
        -------
        G : MultiDiGraph
        """

        G = nx.MultiDiGraph()
        for start in self.connections:
            ends = self.connections[start]
            for end in ends:
                G.add_edge(start, end)

        return G

    @staticmethod
    def __from_repr__(rpr: str):
        import ast
        import importlib
        import torch.nn
        import nord.neural_nets.layers
        d = NeuralDescriptor()
        data = ast.literal_eval(rpr)

        layers_repr = data['Layers']
        layers_repr = layers_repr.replace('<class ', '')
        layers_repr = layers_repr.replace('>', '')
        layers = ast.literal_eval(layers_repr)

        for layer_name in layers:
            layer_class, params = layers[layer_name]
            actual_class = eval(layer_class)
            d.add_layer(actual_class, params, name=layer_name)

        connections_repr = data['Connections']
        connections_repr = connections_repr.replace('<class ', '')
        connections_repr = connections_repr.replace('>', '')
        connections = ast.literal_eval(connections_repr)
        for from_node in connections:
            to_nodes = connections[from_node]
            for to_node in to_nodes:
                d.connect_layers(from_node, to_node)

        d.first_layer = data['First_Layer']
        d.last_layer = data['Last_Layer']
        return d
