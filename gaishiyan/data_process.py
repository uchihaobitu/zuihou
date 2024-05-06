import pandas as pd
import numpy as np

data = pd.read_csv("sock-shop-data/carts-cpu/1/anomalous.csv")
data.drop(columns=["time"], inplace=True)

data_np = np.array(data.values)

print(data_np.shape)
print(data_np)

np.save("gaishiyan/train_data_sock.npy", data_np)