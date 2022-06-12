import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graphviz import Digraph
from scipy import stats

from nord.neural_nets.differentiable.genotypes import Genotype
from statsmodels.stats.multitest import fdrcorrection
import matplotlib


sns.set_style("white")
sns.set_style("ticks")
sns.set_palette('colorblind')
font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)


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


# acc = 'valid_acc'
# new_list = get_accuracies(acc, './log_2a100.txt')
# two_gpus = []

# for x in range(0, 100, 2):
#     two_gpus.append((new_list[x]+new_list[x+1])/2)

# new_list = get_accuracies(acc, './log.txt')
# plt.plot(two_gpus, label='two_gpus')
# plt.plot(new_list, label='one_gpu')
# plt.ylim([80, 99])
# plt.legend()

def plot_train_accs(title, gpu):
    # FULL TRAIN SINGLE GPU
    new_list = get_accuracies(
        'train_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_{gpu}a100/log.txt')
    plt.plot(new_list, label=title+' train')
    new_list = get_accuracies(
        'valid_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_{gpu}a100/log.txt')
    plt.plot(new_list, label=title+' validation')
    plt.ylim([80, 99])
    plt.title(title)
    plt.legend()


plt.figure(dpi=300)
plot_train_accs('1 gpu', 1)
plt.figure(dpi=300)
plot_train_accs('2 gpu', 2)
plt.figure(dpi=300)
plot_train_accs('4 gpu', 4)


plt.figure(dpi=300)
new_list_1 = get_accuracies(
    'train_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_1a100/log.txt')
plt.plot(new_list_1, label='1 gpu', linestyle='--')
new_list_2 = get_accuracies(
    'train_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_2a100/log.txt')
plt.plot(new_list_2, label='2 gpu', linestyle='-')
new_list_4 = get_accuracies(
    'train_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_4a100/log.txt')
plt.plot(new_list_4, label='4 gpu', linestyle='-.')
plt.ylim([90, 100])
plt.legend()
plt.title('Train Accuracy')


plt.figure(dpi=300)
new_list_1 = get_accuracies(
    'valid_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_1a100/log.txt')
plt.plot(new_list_1, label='1 gpu', linestyle='--')
new_list_2 = get_accuracies(
    'valid_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_2a100/log.txt')
plt.plot(new_list_2, label='2 gpu', linestyle='-')
new_list_4 = get_accuracies(
    'valid_acc', f'/home/george/Documents/nord_github/nord/logs/with_reductions/evals/eval-nccl_1node_4a100/log.txt')
plt.plot(new_list_4, label='4 gpu', linestyle='-.')
plt.ylim([80, 100])
plt.legend()
plt.title('Validation Accuracy')


from_index = 10

new_list_1 = new_list_1[from_index:]
new_list_2 = new_list_2[from_index:]
new_list_4 = new_list_4[from_index:]

# validation accs
print('ks 1 vs 2: ', stats.ks_2samp(new_list_1, new_list_2))
print('ks 1 vs 4: ', stats.ks_2samp(new_list_1, new_list_4))
print('ks 2 vs 4: ', stats.ks_2samp(new_list_4, new_list_2))


k_1_2 = stats.ks_2samp(new_list_1, new_list_2)[1]
k_1_4 = stats.ks_2samp(new_list_1, new_list_4)[1]
k_2_4 = stats.ks_2samp(new_list_4, new_list_2)[1]


plt.figure()
sns.kdeplot(np.array(new_list_1)-np.array(new_list_2), label='one vs 2  gpu')
sns.kdeplot(np.array(new_list_1)-np.array(new_list_4), label='one vs 4 gpu')
sns.kdeplot(np.array(new_list_2)-np.array(new_list_4), label='two vs 4 gpu')
# plt.ylim([80, 99])
plt.legend()
plt.title('Validation differences')

print('kruskal 1 vs 2 vs 4: ', stats.kruskal(
    new_list_1, new_list_2, new_list_4))

plt.figure()
df = pd.DataFrame(
    {'one_gpu': new_list_1, 'two_gpu': new_list_2, 'four_gpu': new_list_4})
sns.boxplot(data=df, )
plt.title('')

# print('r 1 vs 2: ', stats.pearsonr(new_list_1, new_list_2))
# print('r 2 vs 4: ', stats.pearsonr(new_list_4, new_list_2))
# print('r 1 vs 4: ', stats.pearsonr(new_list_1, new_list_4))


# def plot_all_genotypes(filename):
#     new_genotypes = get_genotypes(filename)

#     for i in range(len(new_genotypes)):
#         genotype = eval(new_genotypes[i])
#         plot(genotype.normal, f"./genotypes/{filename}_{i}_normal")
#         plot(genotype.reduce, f"./genotypes/{filename}_{i}_reduction")
fdrcorrection((k_1_2, k_1_4, k_2_4))
