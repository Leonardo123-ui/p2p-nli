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
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_graph_pairs(graph_pairs, file_path):
    graphs = [graph for pair in graph_pairs for graph in pair]
    dgl.save_graphs(file_path, graphs)
    logging.info("Saved pair graph at %s", file_path)


def load_graph_pairs(file_path, num_pairs):
    graphs, _ = dgl.load_graphs(file_path)
    graph_pairs = [(graphs[i * 2], graphs[i * 2 + 1]) for i in range(num_pairs)]
    logging.info("Loaded from %s", file_path)
    return graph_pairs


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
            logging.info("Loaded embeddings from %s", file_path)

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


import random


class RSTDataset(Dataset):
    def __init__(
        self,
        rst_path,
        lexical_chains_path,
        embedding_file,
        graph_pair_path,
        batch_size=128,
        subset_ratio=1 / 20,  # 默认选取 1/20 的数据
    ):
        self.rst_path = rst_path
        self.lexical_chains_path = lexical_chains_path
        self.embedding_file = embedding_file
        self.graph_pair_path = graph_pair_path
        self.batch_size = batch_size

        self.data, self.label_encoder = self.load_data()
        self.all_embeddings = load_all_embeddings(self.embedding_file)
        # 随机抽取子集
        if subset_ratio < 1.0:
            total_len = len(self.data)
            logging.info("total len %s", total_len)
            subset_len = int(total_len * subset_ratio)
            logging.info("subset len %s", subset_len)
            subset_indices = random.sample(range(total_len), subset_len)
            self.data = [self.data[i] for i in subset_indices]  # 抽取子集
            self.all_embeddings = [
                self.all_embeddings[i] for i in subset_indices
            ]  # 抽取子集
        self.graph_pairs = []
        # self.load_graph_pairs_in_batches()

    def load_graph_pairs_in_batches(self):
        num_batches = len(self.data) // self.batch_size + (
            1 if len(self.data) % self.batch_size != 0 else 0
        )
        for i in range(num_batches):
            batch_start = i * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(self.data))
            batch_num = i
            self.process_and_save_graph_pairs(batch_start, batch_end, batch_num)

    def process_and_save_graph_pairs(self, batch_start, batch_end, batch_num):
        graph_pairs = []
        # embeddings_batch = load_embeddings_from_directory(
        #     self.embedding_file, limit=(batch_end - batch_start)
        # )
        for idx in range(batch_start, batch_end):
            logging.info("idx: %s", idx)
            rst_result, _ = self.data[idx]
            node_features_premise = extract_node_features(
                self.all_embeddings, idx, "premise"
            )
            rst_relations_premise = rst_result["rst_relation_premise"]
            node_types_premise = rst_result["pre_node_type"]
            g_premise = build_graph(
                node_features_premise, node_types_premise, rst_relations_premise
            )

            node_features_hypothesis = extract_node_features(
                self.all_embeddings, idx, "hypothesis"
            )
            rst_relations_hypothesis = rst_result["rst_relation_hypothesis"]
            node_types_hypothesis = rst_result["hyp_node_type"]
            g_hypothesis = build_graph(
                node_features_hypothesis,
                node_types_hypothesis,
                rst_relations_hypothesis,
            )

            graph_pairs.append((g_premise, g_hypothesis))
        # file_name = self.graph_pair_path + f"_{batch_num}.bin"
        # save_graph_pairs(graph_pairs, file_path=file_name)
        self.graph_pairs.extend(graph_pairs)

    def load_data(self):
        data = []
        rst_results = []
        labels = []

        with open(self.rst_path, "r") as file:
            for line in file:
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
                labels.append(rst_dict["label"])

        logging.info("Got stored RST results")

        lexical_chains = []
        file_paths = []
        if os.path.isfile(self.lexical_chains_path):
            with open(self.lexical_chains_path, "rb") as f:
                lexical_chains = pickle.load(f)
        else:
            for filename in os.listdir(self.lexical_chains_path):
                if filename.endswith(".pkl"):
                    file_path = os.path.join(self.lexical_chains_path, filename)
                    file_paths.append(file_path)
            file_paths.sort(key=lambda x: int(re.search(r"\d+", x).group()))
            for file_path in file_paths:
                lexical_chain = pickle.load(open(file_path, "rb"))
                lexical_chains.extend(lexical_chain)
                logging.info("Got lexical chains from %s", file_path)

        # logging.info(lexical_chains[0])
        # logging.info(type(lexical_chains[0]))

        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        for rst_result, lexical_chain in zip(rst_results, lexical_chains):
            data.append((rst_result, lexical_chain))

        return data, label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # g_premise, g_hypothesis = self.graph_pairs[idx]
        # 动态计算图对
        rst_result, lexical_chain = self.data[idx]
        node_features_premise = extract_node_features(
            self.all_embeddings, idx, "premise"
        )
        rst_relations_premise = rst_result["rst_relation_premise"]
        node_types_premise = rst_result["pre_node_type"]
        g_premise = build_graph(
            node_features_premise, node_types_premise, rst_relations_premise
        )

        node_features_hypothesis = extract_node_features(
            self.all_embeddings, idx, "hypothesis"
        )
        rst_relations_hypothesis = rst_result["rst_relation_hypothesis"]
        node_types_hypothesis = rst_result["hyp_node_type"]
        g_hypothesis = build_graph(
            node_features_hypothesis,
            node_types_hypothesis,
            rst_relations_hypothesis,
        )

        label = rst_result["label"]
        label = self.label_encoder.transform([label])[0]
        return g_premise, g_hypothesis, lexical_chain, label


def create_dataloader(
    rst_path,
    lexical_chains_path,
    embedding_file,
    save_graph_pair,
    batch_size=128,
    subset_ratio=1 / 20,  # 添加子集比例参数
    shuffle=True,
):
    dataset = RSTDataset(
        rst_path,
        lexical_chains_path,
        embedding_file,
        save_graph_pair,
        batch_size=batch_size,
        subset_ratio=subset_ratio,  # 传入子集比例
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
