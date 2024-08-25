import json
import torch
import pickle
import dgl
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from dgl.dataloading import GraphDataLoader
from build_base_graph import build_graph


def save_graph_pairs(graph_pairs, file_path):
    graphs = [graph for pair in graph_pairs for graph in pair]
    dgl.save_graphs(file_path, graphs)
    print("saved pair graph at", file_path)


def load_graph_pairs(file_path, num_pairs):
    graphs, _ = dgl.load_graphs(file_path)
    graph_pairs = [(graphs[i * 2], graphs[i * 2 + 1]) for i in range(num_pairs)]
    print("load from ", file_path)
    return graph_pairs


def extract_node_features(embeddings_data, idx, prefix):
    node_features = {}
    for item in embeddings_data[idx][prefix]:
        node_id, embedding = item
        node_features[node_id] = embedding
    return node_features


def load_embeddings_from_directory(directory_path):
    embeddings_list = []
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):

        if filename.endswith(".npz"):
            # 构建完整的文件路径
            file_path = os.path.join(directory_path, filename)
            print("emb file path:", file_path)
            # 使用 torch.load 加载每个文件中的嵌入
            embeddings = torch.load(file_path)
            embeddings_list.extend(embeddings)

    return embeddings_list


# 定义 RSTDataset 类
class RSTDataset(Dataset):
    def __init__(self, rst_path, lexical_chains_path, embedding_file, graph_pair_path):
        self.rst_path = rst_path
        self.lexical_chains_path = lexical_chains_path
        # self.node_embeddings = torch.load(embedding_file)
        self.node_embeddings = load_embeddings_from_directory(embedding_file)
        self.data, self.label_encoder = self.load_data()
        self.graph_pair_path = graph_pair_path
        if not os.path.exists(self.graph_pair_path):
            self.save_all_graph_pairs()
        self.graph_pairs = load_graph_pairs(self.graph_pair_path, len(self.data))

    def save_all_graph_pairs(self):
        graph_pairs = []
        for idx in range(len(self.data)):
            rst_result, _ = self.data[idx]
            node_features_premise = extract_node_features(
                self.node_embeddings, idx, "premise"
            )
            rst_relations_premise = rst_result["rst_relation_premise"]
            node_types_premise = rst_result["pre_node_type"]
            g_premise = build_graph(
                node_features_premise, node_types_premise, rst_relations_premise
            )

            node_features_hypothesis = extract_node_features(
                self.node_embeddings, idx, "hypothesis"
            )
            rst_relations_hypothesis = rst_result["rst_relation_hypothesis"]
            node_types_hypothesis = rst_result["hyp_node_type"]
            g_hypothesis = build_graph(
                node_features_hypothesis,
                node_types_hypothesis,
                rst_relations_hypothesis,
            )

            graph_pairs.append((g_premise, g_hypothesis))

        save_graph_pairs(graph_pairs, self.graph_pair_path)

    def load_data(self):
        data = []
        rst_results = []
        labels = []

        # 读取 RST 数据
        with open(self.rst_path, "r") as file:
            for line in file:
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
                labels.append(rst_dict["label"])

        print("Got stored RST results")

        # 读取词汇链数据
        if os.path.isfile(self.lexical_chains_path):
            with open(self.lexical_chains_path, "rb") as f:
                lexical_chains = pickle.load(f)
        elif os.path.isdir(self.lexical_chains_path):
            for filename in os.listdir(self.lexical_chains_path):
                if filename.endswith(".pkl"):
                    file_path = os.path.join(self.lexical_chains_path, filename)
                    data = pickle.load(open(file_path, "rb"))
                    lexical_chains.append(data)
                    print("get lexical chains from", file_path)
        # 标签编码
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        for rst_result, lexical_chain in zip(rst_results, lexical_chains):
            data.append((rst_result, lexical_chain))

        return data, label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g_premise, g_hypothesis = self.graph_pairs[idx]
        rst_result, lexical_chain = self.data[idx]
        label = rst_result["label"]  # 标签在 rst_result 中，为字符串形式
        label = self.label_encoder.transform([label])[0]  # 转换为数值形式
        return g_premise, g_hypothesis, lexical_chain, label


# 定义 create_dataloader 函数
def create_dataloader(
    rst_path,
    lexical_chains_path,
    embedding_file,
    save_graph_pair,
    batch_size=32,
    shuffle=True,
):
    dataset = RSTDataset(rst_path, lexical_chains_path, embedding_file, save_graph_pair)
    dataloader = GraphDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


# 定义 collate_fn 函数
def collate_fn(batch):
    g_premises, g_hypotheses, lexical_chains, labels = zip(*batch)
    return (
        list(g_premises),
        list(g_hypotheses),
        list(lexical_chains),
        torch.tensor(labels),
    )


# 示例调用
if __name__ == "__main__":
    rst_path = "path/to/rst_file.json"
    lexical_chains_path = "path/to/lexical_chains.pkl"
    embedding_file = "path/to/embeddings.pt"

    dataloader = create_dataloader(rst_path, lexical_chains_path, embedding_file)
    for batch in dataloader:
        g_premises, g_hypotheses, lexical_chains, labels = batch
        print(labels)
