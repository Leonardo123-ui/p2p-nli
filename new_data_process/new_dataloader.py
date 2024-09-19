import json
import torch
import pickle
import dgl
import os
import re
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from dgl.dataloading import GraphDataLoader
from build_base_graph import build_graph


def merge_files_to_single_pkl(pkl_file_paths, npz_file_paths, output_pkl, output_npz):
    # 按文件名中的数字顺序对 .pkl 文件排序
    sorted_pkl_paths = sorted(
        pkl_file_paths, key=lambda x: int(re.search(r"\d+", x).group())
    )

    # 按文件名中的数字顺序对 .npz 文件排序
    sorted_npz_paths = sorted(
        npz_file_paths, key=lambda x: int(re.search(r"\d+", x).group())
    )

    # 合并 .pkl 文件中的数据
    all_pkl_data = []
    for file_path in sorted_pkl_paths:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            all_pkl_data.extend(data)  # 合并所有 .pkl 数据
        print(f"Loaded and merged .pkl data from {file_path}")

    # 将合并后的 .pkl 数据保存到一个新文件
    with open(output_pkl, "wb") as f:
        pickle.dump(all_pkl_data, f)
    print(f"All .pkl data saved to {output_pkl}")

    # 合并 .npz 文件中的数据
    all_npz_data = []
    for file_path in sorted_npz_paths:
        data = torch.load(file_path)
        all_npz_data.extend(data)  # 合并所有 .npz 数据
        print(f"Loaded and merged .npz data from {file_path}")

    # 将合并后的 .npz 数据保存到一个新文件
    torch.save(all_npz_data, output_npz)
    print(f"All .npz data saved to {output_npz}")


def extract_node_features(embeddings_data, idx, prefix):
    node_features = {}
    for item in embeddings_data[idx][prefix]:
        node_id, embedding = item
        node_features[node_id] = embedding
    return node_features


def load_all_embeddings(directory_path):
    embeddings_list = []

    # 获取文件名列表
    filenames = os.listdir(directory_path)

    # 定义一个函数来提取文件名中的数字
    def extract_number(filename):
        # 使用正则表达式提取文件名中的数字部分
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else float("inf")

    # 根据文件名中的数字进行排序
    sorted_filenames = sorted(filenames, key=extract_number)

    for filename in sorted_filenames:
        if filename.endswith(".npz"):
            file_path = os.path.join(directory_path, filename)
            embeddings = torch.load(file_path)
            embeddings_list.extend(embeddings)
            print("Loaded embeddings from", file_path)

    return embeddings_list


def load_embeddings_from_directory(directory_path, limit=None):
    embeddings_list = []
    count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(directory_path, filename)
            embeddings = torch.load(file_path)
            embeddings_list.extend(embeddings)
            count += 1
            if limit and count >= limit:
                break
    return embeddings_list


class RSTDataset(Dataset):
    def __init__(
        self,
        rst_path,
        lexical_chains_path,
        embedding_path,
        graph_pair_path,
        batch_size=128,
        cache_size=100,
    ):
        self.rst_path = rst_path
        self.lexical_chains_path = lexical_chains_path
        self.embedding_path = embedding_path
        self.graph_pair_path = graph_pair_path
        self.batch_size = batch_size
        self.cache_size = cache_size  # 缓存的数量

        # 加载 RST 数据及标签编码器
        self.rst_results, self.labels = self.load_rst_data()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

        # 收集所有文件路径和计数
        self.lexical_chain_files = self._collect_files(self.lexical_chains_path, ".pkl")
        self.embedding_files = self._collect_files(self.embedding_path, ".npz")
        self.lexical_chain_counts = self._calculate_file_data_counts(
            self.lexical_chain_files
        )
        self.embedding_counts = self._calculate_file_data_counts(self.embedding_files)

        # 用于缓存最近访问的数据块
        self.lexical_chain_cache = {}
        self.embedding_cache = {}

    def _calculate_file_data_counts(self, files):
        """计算每个文件中包含的条目数量，假设每个文件是序列化的数据列表。"""
        counts = []
        for file_path in files:
            with open(file_path, "rb") as f:
                data = (
                    pickle.load(f)
                    if file_path.endswith(".pkl")
                    else torch.load(file_path)
                )
                counts.append(len(data))
        return counts

    def _load_file_chunk(self, idx, files, counts, cache, cache_key):
        """加载文件数据块到缓存中，并根据idx返回对应数据。"""
        if cache_key in cache:
            return cache[cache_key]

        # 清除缓存
        if len(cache) >= self.cache_size:
            cache.clear()

        cumulative_counts = [sum(counts[: i + 1]) for i in range(len(counts))]
        file_idx = next(i for i, count in enumerate(cumulative_counts) if count > idx)

        # 加载整个文件的数据块到缓存
        with open(files[file_idx], "rb") as f:
            data = (
                pickle.load(f)
                if files[file_idx].endswith(".pkl")
                else torch.load(files[file_idx])
            )
        cache[cache_key] = data
        return data

    def _get_data_from_cache(self, idx, files, counts, cache, cache_key):
        """从缓存中获取指定的文件数据。"""
        cumulative_counts = [sum(counts[: i + 1]) for i in range(len(counts))]
        file_idx = next(i for i, count in enumerate(cumulative_counts) if count > idx)
        relative_idx = idx if file_idx == 0 else idx - cumulative_counts[file_idx - 1]

        # 从缓存中加载数据块并返回对应的条目
        data_chunk = self._load_file_chunk(idx, files, counts, cache, cache_key)
        return data_chunk[relative_idx]

    def __getitem__(self, idx):
        # 动态加载 lexical chain 数据块
        lexical_chain = self._get_data_from_cache(
            idx,
            self.lexical_chain_files,
            self.lexical_chain_counts,
            self.lexical_chain_cache,
            "lexical_chain",
        )

        # 动态加载 embedding 数据块
        embedding = self._get_data_from_cache(
            idx,
            self.embedding_files,
            self.embedding_counts,
            self.embedding_cache,
            "embedding",
        )

        # 获取对应的 RST 结果
        rst_result = self.rst_results[idx]

        # 构建图 (premise 和 hypothesis)
        node_features_premise = extract_node_features(embedding, idx, "premise")
        rst_relations_premise = rst_result["rst_relation_premise"]
        node_types_premise = rst_result["pre_node_type"]
        g_premise = build_graph(
            node_features_premise, node_types_premise, rst_relations_premise
        )

        node_features_hypothesis = extract_node_features(embedding, idx, "hypothesis")
        rst_relations_hypothesis = rst_result["rst_relation_hypothesis"]
        node_types_hypothesis = rst_result["hyp_node_type"]
        g_hypothesis = build_graph(
            node_features_hypothesis, node_types_hypothesis, rst_relations_hypothesis
        )

        # 获取并转换标签
        label = rst_result["label"]
        label = self.label_encoder.transform([label])[0]

        return g_premise, g_hypothesis, lexical_chain, label

    def __len__(self):
        return len(self.rst_results)


def create_dataloader(
    rst_path,
    lexical_chains_path,
    embedding_file,
    save_graph_pair,
    batch_size=128,
    shuffle=True,
):
    dataset = RSTDataset(
        rst_path,
        lexical_chains_path,
        embedding_file,
        save_graph_pair,
        batch_size=batch_size,
    )
    dataloader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def collate_fn(batch):
    g_premises, g_hypotheses, lexical_chains, labels = zip(*batch)
    return (
        list(g_premises),
        list(g_hypotheses),
        list(lexical_chains),
        torch.tensor(labels),
    )


if __name__ == "__main__":
    rst_path = "path/to/rst_file.json"
    lexical_chains_path = "path/to/lexical_chains.pkl"
    embedding_file = "path/to/embeddings"
    save_graph_pair = "path/to/save_graph_pairs.bin"

    dataloader = create_dataloader(
        rst_path, lexical_chains_path, embedding_file, save_graph_pair
    )
    for batch in dataloader:
        g_premises, g_hypotheses, lexical_chains, labels = batch
        print(labels)
