import numpy as np
import pandas as pd
import os

class LatentCSVLogger:
    def __init__(self, csv_path, queue_length=10, batch_size=4096):
        """
        csv_path: 存储文件路径
        queue_length: 队列长度（最大批次数）
        batch_size: 每次输入向量数量
        """
        self.csv_path = csv_path
        self.queue_length = queue_length
        self.batch_size = batch_size
        self.max_samples = queue_length * batch_size  # 最大数据量
        
        # 如果文件不存在，创建带表头的空文件
        if not os.path.exists(self.csv_path):
            cols = [f"dim_{i}" for i in range(16)] + ["label"]
            pd.DataFrame(columns=cols).to_csv(self.csv_path, index=False)
    
    def append_batch(self, latent_vectors: np.ndarray, group_ratios: list):
        """
        latent_vectors: np.ndarray, shape=(batch_size,16)
        group_ratios: list, 每个类别占比，和为1, 长度 = num_classes
        """
        assert latent_vectors.shape[0] == self.batch_size, "输入 batch 大小必须等于 batch_size"
        assert latent_vectors.shape[1] == 16, "隐向量维度必须为16"
        assert np.isclose(sum(group_ratios),1), "group_ratios 和必须为1"
        
        # 生成标签
        labels = []
        start_idx = 0
        for cls_idx, ratio in enumerate(group_ratios):
            end_idx = start_idx + int(np.round(ratio * self.batch_size))
            labels.extend([cls_idx] * (end_idx - start_idx))
            start_idx = end_idx
        labels = np.array(labels)
        
        # 如果标签数量不足 batch_size，填充最后一类
        if len(labels) < self.batch_size:
            labels = np.pad(labels, (0, self.batch_size - len(labels)), 'edge')
        elif len(labels) > self.batch_size:
            labels = labels[:self.batch_size]
        
        # 创建 DataFrame
        df_batch = pd.DataFrame(latent_vectors, columns=[f"dim_{i}" for i in range(16)])
        df_batch["label"] = labels
        
        # 读取已有 CSV 数据
        if os.path.exists(self.csv_path):
            df_all = pd.read_csv(self.csv_path)
        else:
            df_all = pd.DataFrame(columns=df_batch.columns)
        
        # 拼接新批次
        df_all = pd.concat([df_all, df_batch], ignore_index=True)
        
        # 如果总量超过最大样本数，丢弃最早 batch_size 数据
        if len(df_all) > self.max_samples:
            df_all = df_all.iloc[self.batch_size:]  # 丢掉最早的 batch
        
        # 写回 CSV（覆盖原文件）
        df_all.to_csv(self.csv_path, index=False)
        # print(f"已写入 {self.batch_size} 条数据，当前总量 {len(df_all)} 条")