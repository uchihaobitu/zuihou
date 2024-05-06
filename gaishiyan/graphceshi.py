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


from tqdm import tqdm

# 这里可以设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = False

# device = torch.device('cuda')
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# X_np = np.load("henon.npy").T
# X_np = np.load("train_use_A2.npy")

output_dir = "../Camrea"
os.makedirs(output_dir, exist_ok=True)  # 使用exist_ok=True避免在目录已存在时抛出异常
numservice = 1
true_graph = 'DAG.gpickle'
data_npy = f'Graph{numservice}_data_norm.npy'
csv_name = f'Graph{numservice}_data_norm.csv'
input_path = f"../../CameraReady-Data/CameraReady-Data/10_services/synthetic/Graph{numservice}/"
data_path = input_path + data_npy
csv_path = input_path + csv_name
graph_path = input_path + true_graph

X_np = np.load(data_path)
X_np = X_np[:300, :]
# X_np = np.load("db_003.npy")
# data_dir = os.path.dirname(data_path)
# print('data_dir',data_dir)
# service, metric = basename(dirname(dirname(data_path))).split("_")
# number = os.path.basename(data_dir)
base_filename = os.path.join(output_dir, f"Graph_{numservice}")
# print('base_filename',base_filename)
X_np = X_np.astype(np.float32)

print(X_np.shape)
print(X_np)

dim = X_np.shape[-1]
GC = np.zeros([dim, dim])
for i in range(dim):
    GC[i, i] = 1
    if i != 0:
        GC[i, i - 1] = 1
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)





full_connect = np.ones(GC.shape)

print(full_connect.shape)
cgru = CRVAE(X.shape[-1], full_connect, hidden=64).to(device=device)
vrae = VRAE4E(X.shape[-1], hidden=64).to(device=device)


train_loss_list = train_phase3(
    cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000, check_every=50, batch_size=128
)  # 0.1


# %%no
GC_est = cgru.GC().cpu().data.numpy()
cor_values = GC_est.flatten()
# 计算保留的边数：总数的15%
num_edges_to_keep = int(len(cor_values) * 0.05)

# 找到前15%的最小值，只有大于或等于这个值的元素会被设置为1
threshold = np.sort(cor_values)[-num_edges_to_keep]
print('threshold',threshold)

# 使用阈值更新原始矩阵：大于等于阈值的设置为1，其他设置为0
result_matrix = np.where(GC_est >= threshold, 1, 0)
np.save(f'{base_filename}.npy', result_matrix)


full_connect = np.load(f'{base_filename}.npy')

cgru = CRVAE(X.shape[-1], full_connect, hidden=64).to(device=device)
vrae = VRAE4E(X.shape[-1], hidden=64).cuda(device=device)
#
#
# train_loss_list = train_phase2(
#     cgru, vrae, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=1000,
#     check_every=50,batch_size=128)
# GC_new = cgru.GC_gai().cpu().data.numpy()
# np.save(f'{base_filename}new.npy', GC_new)



W1 = np.load(f'{base_filename}.npy').T
# W2 = np.load(f'{base_filename}new.npy')
# plot_estimated_graph_v2(W1, W2,numservice)
