import torch
import csv
import os
file_path = "/home/wya/lab_rl/IsaacLab/rsl_rl/rsl_rl/datasets/amp_data_2.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件不存在: {file_path}")

dataset = []
with open(file_path, 'r') as file:
    data = csv.reader(file, delimiter=',')
    # print(list(data)[:2])
    dataset = list(data)
chart_offset = 1
dataset_float = []
for l in dataset[1:]:
    l_float = [float(s) for s in l[:-12]]
    dataset_float.append(l_float)
dataset_t_one_step = torch.tensor(dataset_float, dtype=torch.float32)
dataset_t_last_step = torch.tensor(dataset_float, dtype=torch.float32)
dataset_t = torch.cat([dataset_t_last_step, dataset_t_one_step], dim=1)
print(dataset_t[0])  # 输出数据的形状以确认正确加载