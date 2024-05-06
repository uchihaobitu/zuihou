# import RUN
import argparse
import torch
import pandas as pd
import numpy as np
import json
import networkx as nx
# import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
# import os
# import sys
# import tqdm
from models.utils import pearson_correlation, breaktie

# def Run(datafile):
#     df_data = pd.read_csv(datafile)
#     edges = dict()
    
#     # df_data.drop("timestamp", axis=1, inplace=True) #synthetic don't need this one
#     columns = list(df_data)

#     for c in columns: 
#         # idx = df_data.columns.get_loc(c)
#         edge = RUN.GraphConstruct(c, cuda=cuda, epochs=nrepochs, 
#         lr=learningrate, optimizername=optimizername, file=datafile, args=args)
#         edges.update(edge)
#     return edges, columns

def CreateGraph(edge, columns):
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in edge:
        p1,p2 = pair
        G.add_edge(columns[p2], columns[p1])
    return G

def get_edge_pair(npzfile):
    data = np.load(npzfile).T
    edge_pair = {}
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[i][j] != 0:
                edge_pair[(i,j)] = data[i,j]
    return edge_pair


def main(datafiles, graphfiles,trigger_point, root_cause):
    # edge_pair, columns = Run(datafiles)
    edge_pair = get_edge_pair(graphfiles + args.G_file)
    pruning = pd.read_csv(datafiles)
    columns = pruning.columns.tolist()
    # G = load_pretrain()
    # import pdb
    # pdb.set_trace()
    print('edge_pair', edge_pair)
    G = CreateGraph(edge_pair, columns)

    #画图
    pos = nx.spring_layout(G, k=1.5, iterations=50)  # 增大 k 值和迭代次数
    # node_colors = ['red' if node == 'carts_cpu' else 'lightblue' for node in G.nodes()]
    node_colors = ['red' if node in [trigger_point, root_cause] else 'lightblue' for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight='bold', node_size=700, font_size=9)

    plt.title(f"Graph Visualization for Sample ")
    # plt.figure(figsize=(100, 100))  # 增加图形尺寸
    plt.show()
    plt.savefig(f"Graphv0.png")  # 保存图的可视化到文件
    plt.close()  # 关闭图形，防止在内存中过多积累

    # while not nx.is_directed_acyclic_graph(G):
    #     edge_cor = []
    #     edges = G.edges()
    #     # 使用tqdm包裹迭代器，显示进度条
    #     for edge in tqdm(edges, desc="Processing edges"):
    #         source, target = edge
    #         x = pruning[source].values  # Convert column to numpy array
    #         y = pruning[target].values
    #         edge_cor.append(pearson_correlation(x, y))
    
    #     # Use torch for sorting
    #     tmp = torch.tensor(edge_cor)
    #     tmp_idx = torch.argsort(tmp)
    #     edges = list(edges)
    #     source, target = edges[tmp_idx[0]][0], edges[tmp_idx[0]][1]
    
    #     G.remove_edge(source, target)
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)  # 尝试找到一个环
        if not cycle:
            break  # 如果没有环，退出循环

        edge_cor = []
        # 仅对环中的边进行操作
        for edge in tqdm(cycle, desc="Processing edges"):
            source, target = edge
            x = pruning[source].values  # Convert column to numpy array
            y = pruning[target].values
            edge_cor.append(pearson_correlation(x, y))
        print(edge_cor)

        # 使用torch对相关性进行排序
        tmp = torch.tensor(edge_cor)
        tmp_idx = torch.argsort(tmp)
        # 删除相关性最低的边，从而尝试破坏环
        source, target = cycle[tmp_idx[0]][0], cycle[tmp_idx[0]][1]
        G.remove_edge(source, target)


    # edge_correlations = {}
    # for source, target in tqdm(G.edges):
        # print('source',pruning[source])
        # print('target',pruning[target])
        # edge_correlations[(source, target)] = pearson_correlation(pruning[source], pruning[target])

    # while not nx.is_directed_acyclic_graph(G):
        # print("Graph is not acyclic, remove edge with lowest correlation")
        # 找到相关性最低的边
        # min_cor_edge = min(edge_correlations, key=edge_correlations.get)
        # 移除这条边
        # G.remove_edge(*min_cor_edge)
        # 从字典中移除这条边的相关性
        # del edge_correlations[min_cor_edge]
        # edge_cor = []
        # edges = G.edges()
        # for edge in edges:
        #     source, target = edge
        #     edge_cor.append(pearson_correlation(pruning[source], pruning[target]))
        # tmp = np.array(edge_cor)
        # tmp_idx = np.argsort(tmp)
        # edges = list(edges)
        # source, target= edges[tmp_idx[0]][0], edges[tmp_idx[0]][1]

        # G.remove_edge(source, target)

     # 生成并保存图形的可视化表示
    # plt.figure(figsize=(10, 8))
    node_colors = ['red' if node in [trigger_point, root_cause] else 'lightblue' for node in G.nodes()]

    nx.draw(G,pos, with_labels=True, node_color=node_colors, font_weight='bold', node_size=700, font_size=9)

    plt.title(f"Graph Visualization for Sample ")
    plt.figure(figsize=(100, 100))  # 增加图形尺寸
    plt.show()
    plt.savefig(f"Graphv0.png")  # 保存图的可视化到文件
    plt.close()  # 关闭图形，防止在内存中过多积累
 
    dangling_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    personalization = {}
    for node in G.nodes():
        if node in dangling_nodes:
            personalization[node] = 1.0
        else:
            personalization[node] = 0.5
    pagerank = nx.pagerank(G, personalization=personalization)
    pagerank = dict(sorted(pagerank.items(), key=lambda x: x[1], reverse=True))

    pagerank = breaktie(pagerank, G, trigger_point)
    print(pagerank)
    # import pdb
    # pdb.set_trace()

    pagerank_json = json.dumps(pagerank)

    # 再将JSON字符串写入文件
    with open('pagerank.json', 'w') as json_file:
        json_file.write(pagerank_json)

    if trigger_point != "None":
        for i, node in enumerate(pagerank):
            if node == root_cause:
                if i < 2:
                    print(root_cause, "is in top-1")
                elif i < 4:
                    print(root_cause, "is in top-3")
                elif i < 6:
                    print(root_cause, "is in top-5")
                print(root_cause, "is at", i)
    else:
        previous_score = 0
        to_break = 0
        num_group = 0
        for node, rank in pagerank.items():
            if previous_score == rank:
                num_group += 1
            if previous_score != rank:
                to_break += 1
                to_break += num_group
                num_group = 0
            if node == root_cause:
                if to_break < 2:
                    print(root_cause, "is in top-1")
                elif to_break < 4:
                    print(root_cause, "is in top-3")
                elif to_break < 6:
                    print(root_cause, "is in top-5")
                print(root_cause, "is at", to_break)
            previous_score = rank
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN')

    parser.add_argument('--cuda', type=str, default="cuda:0")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--trigger_point', type=str, default='orders_mem', help='Calculate the distance between node and trigger point')
   
    parser.add_argument('--G_path', type=str, default="../fse_ss_graph/")
    parser.add_argument('--G_file', type=str, default="orders_mem_5new.npy")
    parser.add_argument('--root_path', type=str, default=".")
    parser.add_argument('--data_csv', type=str,default="notime_data.csv")
    # parser.add_argument('--data_path', type=str,default="notime_data.csv")
    parser.add_argument('--data_path', type=str,default="../fse-ss/orders_mem/5/")
    parser.add_argument('--num_workers', type=float, default=10)
    parser.add_argument('--root_cause', type=str,default="orders_mem")

    args = parser.parse_args()

    nrepochs = args.epochs
    learningrate = args.learning_rate
    optimizername = args.optimizer
    cuda=args.cuda
    trigger_point = args.trigger_point
    root_cause = args.root_cause
    graphfiles = args.G_path

    datafiles = args.data_path  + args.data_csv


    main(datafiles, graphfiles, args.trigger_point, args.root_cause)
