# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:00:04 2022

@author: 61995
"""

import os

import numpy as np
import torch
import pandas as pd
import networkx as nx
from tua2 import plot_estimated_graph_v2
import matplotlib.pyplot as plt
from os.path import join, dirname, basename
import pickle
from models.cgru_error import CRVAE, VRAE4E, train_phase1,train_phase2,train_phase3,train_phase4
from pyrca.utils.evaluation import precision, recall, f1, shd
from pyrca.utils.evaluation import shd as SHD
from tqdm import tqdm

# 这里可以设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = False

# device = torch.device('cuda')
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# X_np = np.load("henon.npy").T
# X_np = np.load("train_use_A2.npy")

output_dir = "../Camrea"
numservice = 1
true_graph = 'DAG.gpickle'
data_npy = f'Graph{numservice}_data_norm.npy'
csv_name = f'Graph{numservice}_data_norm.csv'

input_path = f"../../CameraReady-Data/CameraReady-Data/10_services/synthetic/Graph{numservice}/"
data_path = input_path + data_npy
csv_path = input_path + csv_name
graph_path = input_path + true_graph

base_filename = os.path.join(output_dir, f"Graph_{numservice}")


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
    edge_pair = get_edge_pair(f'{base_filename}.npy')
    peuning = pd.read_csv(f'{csv_path}')
    columns = peuning.columns.tolist()
    print(columns)
    # G = load_pretrain()
    # import pdb
    # pdb.set_trace()
    print('edge_pair', edge_pair)
    G = CreateGraph(edge_pair, columns)

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

    #画图
    # pos = nx.spring_layout(G, k=1.5, iterations=50)  # 增大 k 值和迭代次数
    # node_colors = ['red' if node == 'carts_cpu' else 'lightblue' for node in G.nodes()]


    nx.draw(G, with_labels=True, font_weight='bold', node_size=700, font_size=9)

    plt.title(f"Graph Visualization for Sample ")
    plt.figure(figsize=(100, 100))  # 增加图形尺寸
    plt.show()
    plt.savefig(f"Graph{numservice}.png")  # 保存图的可视化到文件
    plt.close()  # 关闭图形，防止在内存中过多积累

    with open(graph_path, 'rb') as f:
        true_G = pickle.load(f)
    print(true_G)

    nx.draw(true_G, with_labels=True, font_weight='bold', node_size=700, font_size=9)

    plt.title(f"Graph Visualization for Sample ")
    plt.figure(figsize=(100, 100))  # 增加图形尺寸
    plt.show()
    # plt.savefig(f"Graph{numservice}.png")  # 保存图的可视化到文件
    plt.close()  # 关闭图形，防止在内存中过多积累

    true_adj_matrix = nx.to_numpy_array(true_G, dtype=int)
    estimated_matrix = nx.to_numpy_array(G, dtype=int)

    # 获取图的节点列表，确保行和列标签一致
    true_nodes = list(true_G.nodes)
    print('true_nodes',true_nodes)
    new_adj_matrix = np.zeros_like(true_adj_matrix)
    for i, node1 in enumerate(columns):
        for j, node2 in enumerate(columns):
            index1 = columns.index(node1)
            index2 = columns.index(node2)
            new_adj_matrix[i, j] = true_adj_matrix[index1, index2]

    new_adj_matrix = pd.DataFrame(new_adj_matrix, index=columns, columns=columns)
    estimated_matrix = pd.DataFrame(estimated_matrix, index=columns, columns=columns)
    print('new_adj_matrix', new_adj_matrix)
    print('estimated_matrix', estimated_matrix)
    # estimated_matrix = pd.DataFrame(W2, index=true_nodes, columns=columns)
    # 打印节点名和邻接矩阵


    adjPrec = precision(new_adj_matrix, estimated_matrix)
    print(f"Precision: {adjPrec:.3f}")
    adjRec = recall(new_adj_matrix, estimated_matrix)
    print(f"Recall: {adjRec:.3f}")
    F1 = f1(new_adj_matrix, estimated_matrix)
    print(f"F1: {F1 :.3f}")
    # shd = SHD(new_adj_matrix, estimated_matrix)
    # print(f"SHD: {shd.get_shd()}")

duqu()
