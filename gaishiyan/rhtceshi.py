from pyrca.analyzers.ht import HT, HTConfig

import numpy as np
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from tua2 import plot_estimated_graph_v2
import matplotlib.pyplot as plt

from os.path import join, dirname, basename
import pickle
from models.cgru_error import CRVAE, VRAE4E, train_phase1,train_phase2,train_phase3,train_phase4

numservice = 1
service = 'orders'
metric = 'cpu'
error_service = 'orders'
error = 'cpu'
input_path = f"../fse-ss/{service}_{metric}/{numservice}/"
graph_path = '../fse_ss_graph/'
graph_1 = f'{service}_{metric}_{numservice}.npy'
graph_2 = f'{service}_{metric}_{numservice}new.npy'
norma_np = 'norm_data.npy'
norma_csv = 'norm_data.csv'
ban_csv = 'notime_data.csv'
yichang_file = 'yichang_notime_data.csv'
normal_data = input_path + norma_np
normal_csv = input_path + norma_csv
abnormal_data = input_path + yichang_file
graph_file = graph_path + graph_2
ban_file = input_path + ban_csv

normal_data = np.load(f'{normal_data}')

pruning = pd.read_csv(f'{abnormal_data}')
columns = pruning.columns.tolist()
print('columns',columns)

def get_edge_pair(npzfile):
    data = np.load(npzfile).T
    edge_pair = {}
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[i][j] != 0:
                edge_pair[(i,j)] = data[i,j]
    return edge_pair

def CreateGraph(edge, columns):
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in edge:
        p1,p2 = pair
        G.add_edge(columns[p2], columns[p1])
    return G

def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("The lengths of the input variables must be the same.")
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    n = len(x)

    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_xy = torch.sum(x * y)
    sum_x_sq = torch.sum(x ** 2)
    sum_y_sq = torch.sum(y ** 2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = torch.sqrt((n * sum_x_sq - sum_x ** 2) * (n * sum_y_sq - sum_y ** 2))

    if denominator == 0:
        return 0

    correlation = numerator / denominator
    return correlation.item()  # Convert to Python float


def duqu():
    # edge_pair, columns = Run(datafiles)
    edge_pair = get_edge_pair(f'{graph_file}')
    peuning = pd.read_csv(f'{ban_file}')
    columns = peuning.columns.tolist()
    print(columns)
    # G = load_pretrain()
    # import pdb
    # pdb.set_trace()
    print('edge_pair', edge_pair)
    G = CreateGraph(edge_pair, columns)

    #画图
    # pos = nx.spring_layout(G, k=1.5, iterations=50)  # 增大 k 值和迭代次数
    # node_colors = ['red' if node == 'carts_cpu' else 'lightblue' for node in G.nodes()]




    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)  # 尝试找到一个环
        if not cycle:
            break  # 如果没有环，退出循环

        edge_cor = []
        # 仅对环中的边进行操作
        for edge in tqdm(cycle, desc="Processing edges"):
            source, target = edge
            x = peuning[source].values  # Convert column to numpy array
            y = peuning[target].values
            edge_cor.append(pearson_correlation(x, y))
        print(edge_cor)

        # 使用torch对相关性进行排序
        tmp = torch.tensor(edge_cor)
        tmp_idx = torch.argsort(tmp)
        # 删除相关性最低的边，从而尝试破坏环
        source, target = cycle[tmp_idx[0]][0], cycle[tmp_idx[0]][1]
        G.remove_edge(source, target)
    nx.draw(G, with_labels=True, font_weight='bold', node_size=700, font_size=9)

    plt.title(f"Graph Visualization for Sample ")
    plt.figure(figsize=(100, 100))  # 增加图形尺寸
    plt.show()
    # plt.savefig(f"Graph{numservice}.png")  # 保存图的可视化到文件
    plt.close()  # 关闭图形，防止在内存中过多积累

    return nx.to_numpy_array(G, dtype=int)




# normal_data = tot_data[:training_samples]
normal_data_df = pd.DataFrame(normal_data, columns=columns)
print('norm',normal_data_df)
abnormal_data_df = pd.read_csv(f'{abnormal_data}')
# abnormal_data_df = pd.DataFrame(abnormal_data, columns=columns)
print('abnorm',abnormal_data_df)

# model = PC(PC.config_class())
# estimated_matrix = model.train(normal_data_df)
estimated_matrix = duqu()
estimated_matrix = pd.DataFrame(estimated_matrix, index=columns, columns=columns)
print(estimated_matrix)

model = HT(config=HTConfig(graph=estimated_matrix))
model.train(normal_data_df)

results = model.find_root_causes(abnormal_data_df, f"{error_service}_{error}", True).to_list()
print(results)