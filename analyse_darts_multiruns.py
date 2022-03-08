
from turtle import color

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from dateutil import parser
from graphviz import Digraph

from nord.neural_nets.differentiable.genotypes import Genotype


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


def genotype_to_networkx(genotype):

    g = nx.DiGraph()
    g.add_node("c_{k-2}")
    g.add_node("c_{k-1}")
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.add_node(str(i))

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
            g.add_edge(u, v, label=op)
    return g


def plot_all_genotypes(filename):
    new_genotypes = get_genotypes(filename)

    for i in range(len(new_genotypes)):
        genotype = eval(new_genotypes[i])
        plot(genotype.normal,
             f"./genotypes/{filename}_{i}_normal")
        plot(genotype.reduce,
             f"./genotypes/{filename}_{i}_reduction")


def plot_last_genotypes(filename, name):
    new_genotypes = get_genotypes(filename)
    genotype = eval(new_genotypes[-1])
    plot(genotype.normal, f"./genotypes/{name}_last_normal")
    plot(genotype.reduce, f"./genotypes/{name}_last_reduction")

    return genotype


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


def get_multi_gpu_distributions(acc, linestyle, label, color, gpus):
    metrics = []
    avg_time = 0
    best = 0
    best_no = None
    best_genotype = None
    for i in range(1, 5):
        filename = f'./logs/with_reductions/nccl_1node_{gpus}a100_{i}/log.txt'
        list = get_accuracies(acc, filename)
        time = get_total_time(filename)
        # print(filename, time)
        this_gen = plot_last_genotypes(filename, f'nccl_1node_{gpus}a100_{i}')
        avg_time += time
        list2 = []
        for j in range(0, len(list), gpus):
            avg = 0
            for k in range(gpus):
                avg += list[k+j]
            avg /= gpus
            list2.append(avg)
        if list2[-1] > best:
            best = list2[-1]
            best_genotype = this_gen
            best_no = i
        if i == 1:
            plt.plot(list2, label=f'{label}',
                     linestyle=linestyle, color=color)
        else:
            plt.plot(list2, linestyle=linestyle, color=color)
        metrics.append(list2)
    avg_time /= 5
    print(f'{gpus} gpu times: {avg_time}')
    print(f'{gpus} gpu best: {best_no} {best} {best_genotype}')
    plt.ylim([80, 100])
    plt.legend()
    return metrics, avg_time


def get_multi_gpu_genotypes(gpus):
    metrics = []
    avg_time = 0
    normals = []
    reductions = []
    for i in range(1, 5):
        filename = f'./logs/with_reductions/nccl_1node_{gpus}a100_{i}/log.txt'
        gens = get_genotypes(filename)
        gen = eval(gens[-1])
        normals.append(genotype_to_networkx(gen.normal))
        reductions.append(genotype_to_networkx(gen.reduce))

    return normals, reductions


def get_ged_values(genotypes):
    values = []
    n = len(genotypes)
    for i in range(n):
        for j in range(i+1, n):
            v = nx.graph_edit_distance(
                genotypes[i], genotypes[j], edge_match=lambda a, b: a['label'] == b['label'])
            values.append(v)
    return values


def get_cross_ged_values(genotypes1, genotypes2):
    values = []
    n = len(genotypes1)
    m = len(genotypes2)
    for i in range(n):
        for j in range(m):
            v = nx.graph_edit_distance(
                genotypes1[i], genotypes2[j], edge_match=lambda a, b: a['label'] == b['label'])
            values.append(v)
    return values


# Plot performance
plt.figure(figsize=(8, 6), dpi=300)
a, time_1 = get_multi_gpu_distributions('valid_acc', ':', '1 gpu', 'C0', 1)
b, time_2 = get_multi_gpu_distributions('valid_acc', '--', '2 gpus', 'C1', 2)
b, time_4 = get_multi_gpu_distributions('valid_acc', '-', '4 gpus', 'C2', 4)
# plt.plot([95 for x in range(50)], linestyle='--')
plt.title('Validation Accuracies')

print('speedup 2:', time_1/time_2)
print('speedup 4:', time_1/time_4)

plt.figure(figsize=(8, 6), dpi=300)
a2 = get_multi_gpu_distributions('train_acc', ':', '1 gpu', 'C0', 1)
b2 = get_multi_gpu_distributions('train_acc', '-', '2 gpus', 'C1', 2)
b2 = get_multi_gpu_distributions('train_acc', '--', '4 gpus', 'C2', 4)

plt.title('train accuracies')

# Get cell similarity

normals_1, reductions_1 = get_multi_gpu_genotypes(1)
normals_1_ged = get_ged_values(normals_1)
reductions_1_ged = get_ged_values(reductions_1)

normals_2, reductions_2 = get_multi_gpu_genotypes(2)
normals_2_ged = get_ged_values(normals_2)
reductions_2_ged = get_ged_values(reductions_2)

normals_4, reductions_4 = get_multi_gpu_genotypes(4)
normals_4_ged = get_ged_values(normals_4)
reductions_4_ged = get_ged_values(reductions_4)

# Get cross similarity
normals_1_2_ged = get_cross_ged_values(normals_1, normals_2)
reductions_1_2_ged = get_cross_ged_values(reductions_1, reductions_2)

normals_1_4_ged = get_cross_ged_values(normals_1, normals_4)
reductions_1_4_ged = get_cross_ged_values(reductions_1, reductions_4)

normals_2_4_ged = get_cross_ged_values(normals_2, normals_4)
reductions_2_4_ged = get_cross_ged_values(reductions_2, reductions_4)

# Plot them
df = pd.DataFrame(
    {'1 vs 4': normals_1_4_ged, '1 vs 2': normals_1_2_ged, '2 vs 4': normals_2_4_ged})
sns.swarmplot(data=df)
plt.title('Cross-Diversity in Normals')
plt.show()

df = pd.DataFrame(
    {'1 vs 4': reductions_1_4_ged, '1 vs 2': reductions_1_2_ged, '2 vs 4': reductions_2_4_ged})
sns.swarmplot(data=df)
plt.title('Cross-Diversity in Reductions')
plt.show()


df = pd.DataFrame(
    {'1 gpu': normals_1_ged, '2 gpus': normals_2_ged, '4 gpus': normals_4_ged})
sns.swarmplot(data=df)
plt.title('Diversity in Normals')
plt.show()

df = pd.DataFrame(
    {'1 gpu': reductions_1_ged, '2 gpus': reductions_2_ged, '4 gpus': reductions_4_ged})
sns.swarmplot(data=df)
plt.title('Diversity in Reductions')
plt.show()


# Plot them
df = pd.DataFrame.from_dict(
    {'1 vs 4': normals_1_4_ged, '1 vs 2': normals_1_2_ged, '2 vs 4': normals_2_4_ged, '1 gpu': reductions_1_ged, '2 gpus': reductions_2_ged, '4 gpus': reductions_4_ged}, orient='index')
df = df.transpose()
sns.boxplot(data=df)
plt.title('Diversity in Normals')
plt.show()

# Plot them
df = pd.DataFrame.from_dict(
    {'1 gpu': reductions_1_ged, '2 gpus': reductions_2_ged, '4 gpus': reductions_4_ged, '1 vs 4': reductions_1_4_ged, '1 vs 2': reductions_1_2_ged, '2 vs 4': reductions_2_4_ged}, orient='index')
df = df.transpose()
sns.swarmplot(data=df)
plt.title('Diversity in Reductions')
plt.show()


print('K-W on normal cells ged')
print(stats.kruskal(normals_1_ged, normals_2_ged, normals_4_ged,
                    normals_1_2_ged, normals_1_4_ged, normals_2_4_ged))


print('K-W on reduction cells ged')
print(stats.kruskal(reductions_1_ged, reductions_2_ged, reductions_4_ged,
                    reductions_1_2_ged, reductions_1_4_ged, reductions_2_4_ged))


# a2 = np.array(a2)
# b2 = np.array(b2)

# single_gpu_train_val_ratio = a/a2
# two_gpu_train_val_ratio = b/b2

# plt.figure()
# plt.plot(single_gpu_train_val_ratio.T, linestyle=':')
# plt.plot(two_gpu_train_val_ratio.T, linestyle='-')
# plt.plot([1 for x in range(50)])
# plt.ylim([0.9, 1.1])
# plt.title('train/validation ratio')
