import networkx as nx
import numpy as np
import torch

# def pearson_correlation(x, y):
#     if len(x) != len(y):
#         raise ValueError("The lengths of the input variables must be the same.")
#     n = len(x)
#     sum_x = sum(x)
#     sum_y = sum(y)
    # sum_xy = sum(x[i] * y[i] for i in range(n))
    # sum_x_sq = sum(x[i] ** 2 for i in range(n))
    # sum_y_sq = sum(y[i] ** 2 for i in range(n))
    # numerator = n * sum_xy - sum_x * sum_y
    # denominator = ((n * sum_x_sq - sum_x ** 2) * (n * sum_y_sq - sum_y ** 2)) ** 0.5
    # if denominator == 0:
    #     return 0
    # correlation = numerator / denominator
    # return correlation

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



def breaktie(pagerank, G, trigger_point):
    if trigger_point == "None":
        return pagerank
    
    rank = []
    tmp_rank = []
    last_score = 0    
    for cnt, (node, score) in enumerate(pagerank.items()):
        if last_score != score:
            if len(tmp_rank) == 0:
                last_score = score
                rank.append(node)
            else:
                ad = []
                for i in range(len(tmp_rank)):
                    try: 
                        distance = nx.shortest_path_length(G, source=trigger_point, target=node)
                    except nx.NetworkXNoPath:
                        distance = 0
                    ad.append(distance)
                ad = np.array(ad)
                # dis_rank = np.argsort(ad, reverse=True)
                dis_rank = np.argsort(ad)[::-1]
                for i in range(len(dis_rank)):
                    rank.append(tmp_rank[dis_rank[i]])
                tmp_rank = [node]
        else:
            tmp_rank.append(node)
            if cnt == len(pagerank)-1:
                ad = []
                for i in range(len(tmp_rank)):
                    try: 
                        distance = nx.shortest_path_length(G, source=trigger_point, target=node)
                    except nx.NetworkXNoPath:
                        distance = 0
                    ad.append(distance)
                ad = np.array(ad)
                # dis_rank = np.argsort(ad, reverse=True)
                dis_rank = np.argsort(ad)[::-1]
                for i in range(len(dis_rank)):
                    rank.append(tmp_rank[dis_rank[i]])
    return rank
    
