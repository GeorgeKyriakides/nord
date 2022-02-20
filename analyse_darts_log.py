import matplotlib.pyplot as plt
from graphviz import Digraph

from nord.neural_nets.differentiable.genotypes import Genotype


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

    g.render(filename, view=False, format='png')


def get_accuracies(text, file_name):
    with open(file_name, "r") as file_read:

        lines = file_read.readlines()

        new_list = []
        idx = 0

        for line in lines:
            if text in line:
                new_list.insert(idx, float(line[:-2].split(' ')[-1]))
                idx += 1
    return new_list


def get_genotypes(file_name):
    text = 'genotype'
    with open(file_name, "r") as file_read:

        lines = file_read.readlines()

        new_list = []
        idx = 0
        for line in lines:
            if text in line:
                new_list.insert(idx, line[35:-1])
                idx += 1
    return new_list


acc = 'valid_acc'
new_list = get_accuracies(acc, './log_2a100.txt')
two_gpus = []

for x in range(0, 100, 2):
    two_gpus.append((new_list[x]+new_list[x+1])/2)

new_list = get_accuracies(acc, './log.txt')
plt.plot(two_gpus, label='two_gpus')
plt.plot(new_list, label='one_gpu')
plt.ylim([80, 99])
plt.legend()


# FULL TRAIN SINGLE GPU
new_list = get_accuracies('train_acc', './logs/train_log_1a100_train.txt')
plt.plot(new_list, label='train')
new_list = get_accuracies('valid_acc', './logs/train_log_1a100_train.txt')
plt.plot(new_list, label='valid')
plt.ylim([80, 99])
plt.legend()


# FULL TRAIN TWO GPU
new_list = get_accuracies('valid_acc', './logs/train_log_1a100_train.txt')
plt.plot(new_list, label='one gpu')
new_list = get_accuracies('valid_acc', './logs/train_log_2a100_train.txt')
plt.plot(new_list, label='two gpu')
plt.ylim([80, 99])
plt.xlim([0, 50])
plt.legend()


def plot_all_genotypes(filename):
    new_genotypes = get_genotypes(filename)

    for i in range(len(new_genotypes)):
        genotype = eval(new_genotypes[i])
        plot(genotype.normal, f"./genotypes/{filename}_{i}_normal")
        plot(genotype.reduce, f"./genotypes/{filename}_{i}_reduction")
