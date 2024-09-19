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


class RSTDataset(Dataset):
    def __init__(
        self,
        rst_path,
        lexical_chains_dir,
        embeddings_dir,
        batch_file_size=2,  # 每批处理的文件数量
    ):
        self.rst_path = rst_path
        self.lexical_chains_dir = lexical_chains_dir
        self.embeddings_dir = embeddings_dir
        self.batch_file_size = batch_file_size

        # 获取所有文件的路径并排序
        self.lexical_files = sorted(
            [f for f in os.listdir(lexical_chains_dir) if f.endswith(".pkl")],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        self.embedding_files = sorted(
            [f for f in os.listdir(embeddings_dir) if f.endswith(".npz")],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )

        assert len(self.lexical_files) == len(self.embedding_files), "文件数不匹配"
        # 用于记录每个文件在 rst_result 中的全局索引偏移量
        self.file_offsets = self.compute_file_offsets()

        self.total_batches = len(self.lexical_files) // self.batch_file_size
        if len(self.lexical_files) % self.batch_file_size != 0:
            self.total_batches += 1

        self.data = self.load_rst_data()
        self.label_encoder = self.get_label_encoder()
        # 存储提前构建好的图，避免在训练时再构建
        self.graph_pairs = []

    def compute_file_offsets(self):
        """
        计算每个文件在 rst_result 中的全局索引偏移量。
        返回一个列表，其中每个元素是 (start_idx, end_idx) 对。
        """
        offsets = []
        current_offset = 0
        for filename in self.lexical_files:
            file_path = os.path.join(self.lexical_chains_dir, filename)
            with open(file_path, "rb") as f:
                lexical_chain = pickle.load(f)
                file_length = len(lexical_chain)  # 获取当前文件的样本数
                offsets.append((current_offset, current_offset + file_length))
                current_offset += file_length
        return offsets

    def load_rst_data(self):
        """
        加载 RST 数据，只需要加载一次。
        """
        rst_results = []
        with open(self.rst_path, "r") as file:
            for line in file:
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
        return rst_results

    def get_label_encoder(self):
        labels = [rst["label"] for rst in self.data]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder

    def load_batch_files(self, batch_num):
        """
        根据批次号加载相应的词汇链和嵌入文件。
        """
        start_idx = batch_num * self.batch_file_size
        end_idx = min((batch_num + 1) * self.batch_file_size, len(self.lexical_files))

        batch_lexical_chains = []
        batch_embeddings = []

        # 加载词汇链
        for i in range(start_idx, end_idx):
            lexical_file = os.path.join(self.lexical_chains_dir, self.lexical_files[i])
            with open(lexical_file, "rb") as f:
                batch_lexical_chains.extend(pickle.load(f))

        # 加载嵌入
        for i in range(start_idx, end_idx):
            embedding_file = os.path.join(self.embeddings_dir, self.embedding_files[i])
            embeddings = torch.load(embedding_file)
            batch_embeddings.extend(embeddings)

        rst_start_idx, rst_end_idx = self.file_offsets[batch_num]
        # 根据词汇链和嵌入构建图
        self.graph_pairs = self.build_graphs(
            batch_lexical_chains, batch_embeddings, rst_start_idx, rst_end_idx
        )

        # return batch_lexical_chains, batch_embeddings

    def build_graphs(self, lexical_chains, embeddings, start_idx, end_idx):
        """
        根据加载的词汇链和嵌入构建图。
        """
        graph_pairs = []
        count = 0  # 计数器，用于batch embedding and lexical chain
        for idx in range(start_idx, end_idx):
            rst_result = self.data[idx]

            # 构建图
            node_features_premise = extract_node_features(embeddings, count, "premise")
            rst_relations_premise = rst_result["rst_relation_premise"]
            node_types_premise = rst_result["pre_node_type"]
            g_premise = build_graph(
                node_features_premise, node_types_premise, rst_relations_premise
            )

            node_features_hypothesis = extract_node_features(
                embeddings, count, "hypothesis"
            )
            rst_relations_hypothesis = rst_result["rst_relation_hypothesis"]
            node_types_hypothesis = rst_result["hyp_node_type"]
            g_hypothesis = build_graph(
                node_features_hypothesis,
                node_types_hypothesis,
                rst_relations_hypothesis,
            )

            graph_pairs.append(
                (g_premise, g_hypothesis, lexical_chains[count], rst_result["label"])
            )
            count += 1

        return graph_pairs

    def __len__(self):
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        g_premise, g_hypothesis, lexical_chain, label = self.graph_pairs[idx]
        label = self.label_encoder.transform([label])[0]
        return g_premise, g_hypothesis, lexical_chain, label


# def create_dataloader(
#     rst_path,
#     lexical_chains_dir,
#     embedding_dir,
#     batch_file_size=10,  # 每批加载的文件数量
#     shuffle=True,
# ):
#     """
#     按批次加载数据的 dataloader 函数。
#     每次调用时按指定的文件批次加载数据。
#     """
#     dataset = RSTDataset(
#         rst_path,
#         lexical_chains_dir,
#         embedding_dir,
#         batch_file_size=batch_file_size,
#     )

#     return dataset  # 返回的是按批次加载的 dataset，而非 dataloader


# def collate_fn(batch):
#     """
#     用于将一个批次的数据打包在一起。
#     """
#     g_premises, g_hypotheses, lexical_chains, labels = zip(*batch)
#     return (
#         list(g_premises),
#         list(g_hypotheses),
#         list(lexical_chains),
#         torch.tensor(labels),
#     )


# if __name__ == "__main__":
#     rst_path = "path/to/rst_file.json"
#     lexical_chains_path = "path/to/lexical_chains.pkl"
#     embedding_file = "path/to/embeddings"
#     save_graph_pair = "path/to/save_graph_pairs.bin"

#     dataloader = create_dataloader(
#         rst_path, lexical_chains_path, embedding_file, save_graph_pair
#     )
#     for batch in dataloader:
#         g_premises, g_hypotheses, lexical_chains, labels = batch
#         print(labels)
