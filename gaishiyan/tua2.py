import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

output_la = "../picture"


def plot_estimated_graph_v2(W_est, W,serve_number):
    adj_matrix = W_est
    estimated_matrix = W

    # 计算差异矩阵
    difference_matrix = adj_matrix - estimated_matrix
    base_filename = os.path.join(output_la, f"Graph_{serve_number}")

    number = 0
    dim = difference_matrix.shape[-1]
    for i in range(dim):
        for j in range(dim):
            if difference_matrix[i, j] != 0:
                number += 1
                print(difference_matrix[i, j])
    print(number)

    # 定义色彩映射：中间为白色，两边为红蓝
    cdict = {
        'red':   [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.3, 0.3)],
        'blue':  [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.2, 0.2)]
    }
    custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)

    # 绘制热图
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 真实邻接矩阵
    cax = ax[0].matshow(adj_matrix, cmap=custom_cmap, vmin=-2, vmax=2)
    fig.colorbar(cax, ax=ax[0])
    ax[0].set_title('dagma')

    # 估计的邻接矩阵
    cax = ax[1].matshow(estimated_matrix, cmap=custom_cmap, vmin=-2, vmax=2)
    fig.colorbar(cax, ax=ax[1])
    ax[1].set_title('Diff-TS')

    # 差异矩阵
    cax = ax[2].matshow(difference_matrix, cmap=custom_cmap, vmin=-2, vmax=2)
    fig.colorbar(cax, ax=ax[2])
    ax[2].set_title('Difference matrix')


    plt.savefig(f'{base_filename}.png')

    plt.close()  # 关闭图形，防止在内存中过多积累


# 示例数据
if __name__ == '__main__':
    W_est = np.load('carts_cpu_1.npy')
    print((W_est > 0.55).astype(int))
    # W_est = (W_est > 0.55).astype(int)
    dim = W_est.shape[-1]

    # dim = W_est.shape[-1]
    # for i in range(dim):
    #     for j in range(dim):
    #         if W_est[i,j] != 0:
    #             number += 1
    #             print(W_est[i,j])
    # print(number)
    # W = np.load('GC_henon_A2new.npy')

    # 随机分配一些值，确保它们在-5到5范围内
    # indexes = np.random.choice(np.arange(571*571), 1800, replace=False)
    # W.ravel()[indexes] = np.random.uniform(-5, 5, 1800)  # 使用均匀分布来模拟数据
    # W = np.zeros([dim,dim])
    W = np.load('carts_cpu_1new.npy')
    plot_estimated_graph_v2(W_est, W)
