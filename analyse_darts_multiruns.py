from dateutil import parser
import numpy as np
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

    g.render(filename, view=False)


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


def get_total_time(file_name):
    start_time = 0
    end_time = 0
    with open(file_name, "r") as file_read:

        lines = file_read.readlines()

        new_list = []
        idx = 0

        for line in lines:
            if 'epoch 0' in line:
                start_time = line.split(',')[0]
            end_time = line.split(',')[0]
    return (parser.parse(end_time) - parser.parse(start_time)).total_seconds()


def get_single_gpu_distributions(acc, linestyle, label):
    metrics = []
    for i in range(1, 6):
        list = get_accuracies(acc, f'./logs/log_1a100_{i}.txt')
        metrics.append(list)
        time = get_total_time(f'./logs/log_1a100_{i}.txt')
        plt.plot(list, label=f'{label}_{i} ', linestyle=linestyle)
    plt.ylim([80, 99])
    plt.legend()
    return metrics


def get_multi_gpu_distributions(acc, linestyle, label, gpus):
    metrics = []
    for i in range(1, 2):
        list = get_accuracies(acc, f'./logs/log_{gpus}a100_async_{i}.txt')
        time = get_total_time(f'./logs/log_{gpus}a100_async_{i}.txt')
        list2 = []
        for i in range(0, len(list), gpus):
            avg = 0
            for j in range(gpus):
                avg += list[i+j]
            avg /= gpus
            list2.append(avg)
        plt.plot(list2, label=f'{label}_{i} {time} secs', linestyle=linestyle)
        metrics.append(list2)
    plt.ylim([80, 99])
    plt.legend()
    return metrics


def get_multi_gpu_distributions2(acc, linestyle, label, gpus):
    metrics = []
    for i in range(1, 2):
        list = get_accuracies(acc, f'./logs/log_{gpus}a100_async_{i}.txt')
        time = get_total_time(f'./logs/log_{gpus}a100_async_{i}.txt')
        list2 = []
        for j in range(gpus):
            list2.append([])
            for i in range(j, len(list), gpus):
                list2[j].append(list[i])
            plt.plot(
                list2[j], label=f'{label}_{j} ', linestyle=linestyle)

    plt.ylim([80, 99])
    plt.legend()
    return list2


a = get_single_gpu_distributions('valid_acc', ':', 'single_gpu_')
b = get_multi_gpu_distributions2('valid_acc', '-', 'multi_gpu_', 2)

a = np.array(a)
b = np.array(b)

plt.figure()
a2 = get_single_gpu_distributions('train_acc', ':', 'single_gpu_')
b2 = get_multi_gpu_distributions2('train_acc', '-', 'multi_gpu_', 2)

a2 = np.array(a2)
b2 = np.array(b2)

single_gpu_train_val_ratio = a/a2
two_gpu_train_val_ratio = b/b2

plt.figure()
plt.plot(single_gpu_train_val_ratio.T, linestyle=':')
plt.plot(two_gpu_train_val_ratio.T, linestyle='-')
plt.ylim([0.9, 1.1])
