import os
from copy import deepcopy

from nats_bench import create
from numpy import random

from nord.utils import download_file_from_google_drive, untar_file, DATA_ROOT

NATSBENCH_NAME = "NATS-tss-v1_0-3ffb9-simple"
NATSBENCH_TFRECORD = DATA_ROOT + "/" + NATSBENCH_NAME
file_id = "17_saCsj_krKjlCBLOJEpNtzPXArMCqxU"


OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]


def get_combination(space, num):
    combs = []
    for i in range(num):
        if i == 0:
            for func in space:
                combs.append([(func, i)])
        else:
            new_combs = []
            for string in combs:
                for func in space:
                    xstring = string + [(func, i)]
                    new_combs.append(xstring)
            combs = new_combs
    return combs


class Structure:
    def __init__(self, genotype):
        assert isinstance(genotype, list) or isinstance(
            genotype, tuple
        ), "invalid class of genotype : {:}".format(type(genotype))
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for idx, node_info in enumerate(genotype):
            assert isinstance(node_info, list) or isinstance(
                node_info, tuple
            ), "invalid class of node_info : {:}".format(type(node_info))
            assert len(node_info) >= 1, "invalid length : {:}".format(len(node_info))
            for node_in in node_info:
                assert isinstance(node_in, list) or isinstance(
                    node_in, tuple
                ), "invalid class of in-node : {:}".format(type(node_in))
                assert (
                    len(node_in) == 2 and node_in[1] <= idx
                ), "invalid in-node : {:}".format(node_in)
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(deepcopy(node_info)))

    def tolist(self, remove_str):
        # convert this class to the list, if remove_str is 'none', then remove the 'none' operation.
        # note that we re-order the input node in this function
        # return the-genotype-list and success [if unsuccess, it is not a connectivity]
        genotypes = []
        for node_info in self.nodes:
            node_info = list(node_info)
            node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
            node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
            if len(node_info) == 0:
                return None, False
            genotypes.append(node_info)
        return genotypes, True

    def node(self, index):
        assert index > 0 and index <= len(self), "invalid index={:} < {:}".format(
            index, len(self)
        )
        return self.nodes[index]

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = "|".join([x[0] + "~{:}".format(x[1]) for x in node_info])
            string = "|{:}|".format(string)
            strings.append(string)
        return "+".join(strings)

    def check_valid(self):
        nodes = {0: True}
        for i, node_info in enumerate(self.nodes):
            sums = []
            for op, xin in node_info:
                if op == "none" or nodes[xin] is False:
                    x = False
                else:
                    x = True
                sums.append(x)
            nodes[i + 1] = sum(sums) > 0
        return nodes[len(self.nodes)]

    def to_unique_str(self, consider_zero=False):
        # this is used to identify the isomorphic cell, which rerquires the prior knowledge of operation
        # two operations are special, i.e., none and skip_connect
        nodes = {0: "0"}
        for i_node, node_info in enumerate(self.nodes):
            cur_node = []
            for op, xin in node_info:
                if consider_zero is None:
                    x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                elif consider_zero:
                    if op == "none" or nodes[xin] == "#":
                        x = "#"  # zero
                    elif op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                else:
                    if op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                cur_node.append(x)
            nodes[i_node + 1] = "+".join(sorted(cur_node))
        return nodes[len(self.nodes)]

    def check_valid_op(self, op_names):
        for node_info in self.nodes:
            for inode_edge in node_info:
                # assert inode_edge[0] in op_names, 'invalid op-name : {:}'.format(inode_edge[0])
                if inode_edge[0] not in op_names:
                    return False
        return True

    def __repr__(self):
        return "{name}({node_num} nodes with {node_info})".format(
            name=self.__class__.__name__, node_info=self.tostr(), **self.__dict__
        )

    def __len__(self):
        return len(self.nodes) + 1

    def __getitem__(self, index):
        return self.nodes[index]

    @staticmethod
    def str2structure(xstr):
        if isinstance(xstr, Structure):
            return xstr
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(
            type(xstr)
        )
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(
                    xinput
                )
            inputs = (xi.split("~") for xi in inputs)
            input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
            genotypes.append(input_infos)
        return Structure(genotypes)

    @staticmethod
    def str2fullstructure(xstr, default_name="none"):
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(
            type(xstr)
        )
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(
                    xinput
                )
            inputs = (xi.split("~") for xi in inputs)
            input_infos = list((op, int(IDX)) for (op, IDX) in inputs)
            all_in_nodes = list(x[1] for x in input_infos)
            for j in range(i):
                if j not in all_in_nodes:
                    input_infos.append((default_name, j))
            node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
            genotypes.append(tuple(node_info))
        return Structure(genotypes)

    @staticmethod
    def gen_all(search_space, num, return_ori):
        assert isinstance(search_space, list) or isinstance(
            search_space, tuple
        ), "invalid class of search-space : {:}".format(type(search_space))
        assert (
            num >= 2
        ), "There should be at least two nodes in a neural cell instead of {:}".format(
            num
        )
        all_archs = get_combination(search_space, 1)
        for i, arch in enumerate(all_archs):
            all_archs[i] = [tuple(arch)]

        for inode in range(2, num):
            cur_nodes = get_combination(search_space, inode)
            new_all_archs = []
            for previous_arch in all_archs:
                for cur_node in cur_nodes:
                    new_all_archs.append(previous_arch + [tuple(cur_node)])
            all_archs = new_all_archs
        if return_ori:
            return all_archs
        else:
            return [Structure(x) for x in all_archs]


class NATSBench_Evaluator:
    def __init__(self, filepath=None) -> None:
        if filepath is None:
            filepath = NATSBENCH_TFRECORD
        if not os.path.isdir(DATA_ROOT):
            os.mkdir(DATA_ROOT)

        if not os.path.isdir(filepath):
            print("Downloading NATSBench Data.")
            download_file_from_google_drive(file_id, filepath + ".tar")
            print("Downloaded, extracting.")
            untar_file(filepath + ".tar", DATA_ROOT)

        self.api = create(
            filepath + "/" + NATSBENCH_NAME, "tss", fast_mode=True, verbose=False
        )

    def descriptor_evaluate(
        self, descriptor, metrics=["validation_accuracy", "time_cost"]
    ):

        arch = NATSBench_Evaluator._descriptor_to_nasnet(descriptor)
        index = self.api.query_index_by_arch(arch)

        api_metrics = self.api.get_more_info(arch, dataset="cifar10-valid", hp="200")
        latency = self.api.get_latency(index, dataset="cifar10-valid")
        arch_metrics = {
            "train_loss": api_metrics["train-loss"],
            "validation_loss": api_metrics["valid-loss"],
            "test_loss": api_metrics["test-loss"],
            "train_accuracy": api_metrics["train-accuracy"],
            "validation_accuracy": api_metrics["valid-accuracy"],
            "test_accuracy": api_metrics["test-accuracy"],
            "latency": latency,
            "time_cost": api_metrics["train-all-time"] + api_metrics["valid-per-time"],
        }
        return [arch_metrics[x] for x in metrics]

    @staticmethod
    def _add_node(descriptor, layer, node_no):

        node_name = f"my_node_{node_no:03}"
        incomings = deepcopy(descriptor.incoming_connections[layer])

        descriptor.add_layer("NODE", None, node_name)

        descriptor.connect_layers(node_name, layer)
        for incoming_layer in incomings:
            if "node" not in incoming_layer:
                descriptor.connect_layers(incoming_layer, node_name)
                descriptor.disconnect_layers(incoming_layer, layer)

        return descriptor, node_name

    @staticmethod
    def _get_nasnet_edges(descriptor):

        # Add a node before each layer
        node_no = 0

        descriptor = deepcopy(descriptor)
        original_layers = list(descriptor.layers.keys())
        first_layer = descriptor.first_layer
        original_layers.remove(first_layer)

        new_nodes = []
        for layer in original_layers:
            descriptor, new_node = NATSBench_Evaluator._add_node(
                descriptor, layer, node_no
            )
            new_nodes.append(new_node)
            node_no += 1

        # Group nodes that receive the same input
        # into a single node

        groups = {}
        original_layers.append(first_layer)
        for layer in original_layers:
            connections = descriptor.connections[layer]
            if len(connections) > 0:
                for c in connections:
                    groups[c] = connections[0]

        # Create the OPS edges

        edges = {}
        for node in new_nodes:
            incomings = descriptor.incoming_connections[node]
            for incoming in incomings:
                # incomings (old layers) to the new nodes ('my_node_XX')
                # are replaced with their new nodes parents and the op is saved as the last
                # element of the tuple
                node_incoming = descriptor.incoming_connections[incoming]

                if len(node_incoming) == 0:
                    continue
                node_incoming = groups[node_incoming[0]]

                if node_incoming in edges:
                    edges[node_incoming][groups[node]] = descriptor.layers[incoming]
                else:
                    edges[node_incoming] = {groups[node]: descriptor.layers[incoming]}

        return edges

    @staticmethod
    def _descriptor_to_nasnet(descriptor, max_nodes=4):

        edges = NATSBench_Evaluator._get_nasnet_edges(descriptor)
        if len(edges.keys()) + 1 > max_nodes:
            max_nodes = len(edges.keys()) + 1
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                # node_str = "{:}<-{:}".format(i, j)
                op_name = "none"
                if len(list(edges.keys())) > j:
                    node_name = list(edges.keys())[j]
                    if f"my_node_{i:03}" in edges[node_name]:
                        op_name = edges[node_name][f"my_node_{i:03}"][0]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)


def random_topology_func(op_names, max_nodes=4):
    # Return a random architecture
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                # node_str = "{:}<-{:}".format(i, j)
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    return random_architecture
