import sys
import nord.neural_nets.differentiable.genotypes as genotypes
from graphviz import Digraph


def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20',
                       height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")

genotype = eval("Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))")
plot(genotype.normal, "2gpu-normal")
plot(genotype.reduce, "2gpu-reduction")
