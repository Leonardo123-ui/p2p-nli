import json
import torch
import numpy as np
import nltk
import os
import pickle
import nltk
import glob
from nltk.tokenize import word_tokenize
from transformers import XLMRobertaModel, XLMRobertaTokenizer

# 下载punkt分词器模型（如果还没有下载过）
# nltk.download('punkt')

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.DM_RST import *


def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度
    """
    # 确保向量是二维的，形状为 (1, n)
    v1 = v1.reshape(1, -1)
    v2 = v2.reshape(1, -1)

    # 计算余弦相似度
    dot_product = np.dot(v1, v2.T)  # 使用 v2.T 进行转置
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def extract_node_features(embeddings_data, idx, prefix):
    node_features = {}
    for item in embeddings_data[idx][prefix]:
        node_id, embedding = item
        node_features[node_id] = embedding
    return node_features


def load_all_data(data_processor, model_output_path, original_data_path):
    ori_data = data_processor.read_json_lines(original_data_path)  # 读取原始数据
    train_data = data_processor.load_json(model_output_path)
    rst_results = data_processor.get_rst(train_data, ori_data)
    return train_data, ori_data, rst_results


# 主要是获取处理后的数据，包括rst树的信息，以及节点的字符串表示和bert embeddings，以及词汇链信息
class Data_Processor:
    def __init__(self, mode, save_dir, purpose):
        self.save_dir = os.path.join(save_dir, purpose)
        self.rst_path = "rst_result.jsonl"
        self.save_or_not = mode

    def read_json_lines(self, file_path):
        oridata = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    # 解析每一行为一个字典
                    record = json.loads(line.strip())
                    oridata.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(e)
        return oridata

    def load_json(self, json_path):  # 加载模型输出结果和相关index
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_data_length(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        print("数据长度：", len(data))
        return len(data)

    def write_jsonl(self, path, data):
        with open(path, "w", encoding="utf-8") as file:
            for record in data:
                # 将字典转换为JSON字符串格式
                json_record = json.dumps(record)
                # 将JSON记录写入文件，每个记录后跟一个换行符
                file.write(json_record + "\n")
        print(f"Saved records to {path} successfully.")

    def get_articles_by_mark(self, oridata, news_mark) -> dict:  # 返回原文中的文章
        # 将 "0_1" 这样的标记拆分成 list_index 和 item_index
        list_index, item_index = map(int, news_mark.split("_"))
        # 使用解析出的索引从 data 里获取数据
        return oridata[list_index]["news"][item_index]

    def get_tree(self, a):
        """
        把DM——rst中提取出的node结果变成🌲的类型
        :param a: node_number
        :return:tree树的表示，leaf_node叶子节点的下标，parent_dict父节点的下标和范围
        """
        tree = []
        list_new = [elem for sublist in a for elem in (sublist[:2], sublist[2:])]
        parent_node = [1]  # 根节点的node表示为1

        parent_dict = {}
        leaf_node = []
        for index, i in enumerate(list_new):
            if i[0] == i[1]:
                leaf_node.append(
                    index + 2
                )  # index从0开始，所以算上根节点，树节点的表示应该=index+2
            else:
                parent_node.append(index + 2)
                key = str(i[0]) + "_" + str(i[1])
                parent_dict[key] = index + 2  # 形式为{"1_12":2}
            if index < 2:
                tree.append([1, index + 2])  # 注意这里的层级

        for index, j in enumerate(a):
            if index == 0:
                continue
            else:
                key = str(j[0]) + "_" + str(j[3])
                parent = parent_dict[key]
                tree.append([parent, (index + 1) * 2])
                tree.append([parent, (index + 1) * 2 + 1])
        return parent_dict, leaf_node, tree

    def get_rst(self, data, original_data):  # a 模型输出 b 原始数据
        """
        获取前提和假设的rst🌲 分析结果，包括树节点、节点的string、节点的核性、节点间的关系
        :param data:
        :return:
        """
        rst_results_store_path = os.path.join(
            self.save_dir, self.rst_path
        )  # rst信息的存储路径
        print("the rst result path", rst_results_store_path)
        if os.path.exists(rst_results_store_path):
            rst_results = self.get_stored_rst(rst_results_store_path)
            print("exist rst result")
            return rst_results  # 如果存在 直接读取

        my_rst_tree = RST_Tree()
        model = my_rst_tree.init_model()  # 初始化模型
        precess_rst_tree = precess_rst_result()
        batch_size = 100  # 设置批处理大小
        rst_results = []
        count = 0
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]

            # 批量构建输入句子列表
            input_sentences = []
            for item in batch_data:
                news1_mark = item["news1_mark"]
                news1 = self.get_articles_by_mark(original_data, news1_mark)
                article1 = news1["article"]
                news2_mark = item["news2_mark"]
                news2 = self.get_articles_by_mark(original_data, news2_mark)
                article2 = news2["article"]
                input_sentences.append(article1)
                input_sentences.append(article2)

            # 批量进行推理
            (
                input_sentences_batch,
                all_segmentation_pred_batch,
                all_tree_parsing_pred_batch,
            ) = my_rst_tree.inference(model, input_sentences)

            # new_data = []
            for index, i in enumerate(batch_data):
                segments_pre = precess_rst_tree.merge_strings(
                    input_sentences_batch[index * 2],
                    all_segmentation_pred_batch[index * 2],
                )  # 获取单个edu的string
                segments_hyp = precess_rst_tree.merge_strings(
                    input_sentences_batch[index * 2 + 1],
                    all_segmentation_pred_batch[index * 2 + 1],
                )  # 获取单个edu的string

                if all_tree_parsing_pred_batch[index * 2][0] == "NONE":
                    node_number_pre = 1
                    node_string_pre = [segments_pre]
                    RelationAndNucleus_pre = "NONE"
                    tree_pre = [[1, 1]]
                    leaf_node_pre = [1]
                    parent_dict_pre = {"1_1": 1}
                    print("premise1 no rst")
                else:
                    rst_info_pre = all_tree_parsing_pred_batch[index * 2][
                        0
                    ].split()  # 提取出rst结构，字符串形式

                    node_number_pre, node_string_pre = precess_rst_tree.use_rst_info(
                        rst_info_pre, segments_pre
                    )  # 遍历RST信息，提取关系和标签信息
                    RelationAndNucleus_pre = precess_rst_tree.get_RelationAndNucleus(
                        rst_info_pre
                    )  # 提取核性和关系
                    parent_dict_pre, leaf_node_pre, tree_pre = self.get_tree(
                        node_number_pre
                    )
                if all_tree_parsing_pred_batch[index * 2 + 1][0] == "NONE":
                    node_number_hyp = 1
                    node_string_hyp = [segments_hyp]
                    RelationAndNucleus_hyp = "NONE"
                    tree_hyp = [[1, 1]]
                    leaf_node_hyp = [1]
                    parent_dict_hyp = {"1_1": 1}
                    print("premise2 no rst")
                else:
                    rst_info_hyp = all_tree_parsing_pred_batch[index * 2 + 1][
                        0
                    ].split()  # 提取出rst结构，字符串形式
                    node_number_hyp, node_string_hyp = precess_rst_tree.use_rst_info(
                        rst_info_hyp, segments_hyp
                    )  # 遍历RST信息，提取关系和标签信息
                    RelationAndNucleus_hyp = precess_rst_tree.get_RelationAndNucleus(
                        rst_info_hyp
                    )  # 提取核性和关系
                    parent_dict_hyp, leaf_node_hyp, tree_hyp = self.get_tree(
                        node_number_hyp
                    )

                rst_results.append(
                    {
                        "pre_node_number": node_number_pre,
                        "pre_node_string": node_string_pre,
                        "pre_node_relations": RelationAndNucleus_pre,
                        "pre_tree": tree_pre,
                        "pre_leaf_node": leaf_node_pre,
                        "pre_parent_dict": parent_dict_pre,
                        "hyp_node_number": node_number_hyp,  # 这里的hypothesis其实是premise2
                        "hyp_node_string": node_string_hyp,
                        "hyp_node_relations": RelationAndNucleus_hyp,
                        "hyp_tree": tree_hyp,
                        "hyp_leaf_node": leaf_node_hyp,
                        "hyp_parent_dict": parent_dict_hyp,
                    }
                )
                print(count, "count")
                count += 1  # 增加计数器
                # 每5000条保存一次
                if count % 5000 == 0:
                    os.makedirs(self.save_dir, exist_ok=True)
                    rst_name = str(count) + "_rst_result.jsonl"
                    self.write_jsonl(os.path.join(self.save_dir, rst_name), rst_results)
                    print(
                        f"Saved {len(rst_results)} records to {os.path.join(self.save_dir, self.rst_path)}"
                    )
                    rst_results = []  # 清空列表以便下次使用
        # 循环结束后，保存剩余的结果
        if rst_results and count < 5000:
            os.makedirs(self.save_dir, exist_ok=True)

            self.write_jsonl(os.path.join(self.save_dir, self.rst_path), rst_results)
            print(
                f"Saved {len(rst_results)} records to {os.path.join(self.save_dir, self.rst_path)}"
            )
        elif rst_results:
            os.makedirs(self.save_dir, exist_ok=True)
            rst_name = str(count) + "left_rst_result.jsonl"
            self.write_jsonl(os.path.join(self.save_dir, rst_name), rst_results)
            print(
                f"Saved {len(rst_results)} records to {os.path.join(self.save_dir, self.rst_path)}"
            )

        print(len(rst_results), "最后剩下的rst results length")
        return rst_results

    def get_stored_rst(self, path):
        rst_results = []
        with open(path, "r") as file:
            for line in file:
                # 解析JSON字符串为字典
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
        print("got stored rst result from：", path)
        return rst_results


from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class RSTEmbedder:
    def __init__(self, model_path, save_dir, purpose, save_or_not):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.model = XLMRobertaModel.from_pretrained(model_path)
        self.save_dir_lexical = os.path.join(save_dir, purpose)
        self.save_or_not = save_or_not
        self.lexical_matrix_path = "lexical_matrixes.pkl"

    def write_jsonl(self, path, data):
        with open(path, "w", encoding="utf-8") as file:
            for record in data:
                # 将字典转换为JSON字符串格式
                json_record = json.dumps(record)
                # 将JSON记录写入文件，每个记录后跟一个换行符
                file.write(json_record + "\n")
        print(f"Saved records to {path} successfully.")

    def get_stored_rst(self, paths):
        rst_results = []
        if isinstance(paths, list):
            for path in paths:
                with open(path, "r") as file:
                    for line in file:
                        rst_dict = json.loads(line.strip())
                        rst_results.append(rst_dict)
        elif isinstance(paths, str):
            with open(paths, "r") as file:
                for line in file:
                    rst_dict = json.loads(line.strip())
                    rst_results.append(rst_dict)
                    # if len(rst_results) == 40:
                    #     break
        print("got stored rst result")
        return rst_results

    @staticmethod
    def find_leaf_node(number_list, all_string):
        """
        找到叶子节点的string，以及其在树中的节点表示
        :param number_list:
        :return:
        """
        leaf_node_index = []
        leaf_string = []
        for index, sub_list in enumerate(number_list):
            if sub_list[0] == sub_list[1]:
                leaf_string.append(all_string[index][0])
                leaf_node_index.append(index * 2 + 1)  # 节点从0开始的
            if sub_list[2] == sub_list[3]:
                leaf_string.append(all_string[index][1])
                leaf_node_index.append(index * 2 + 2)
        if len(leaf_string) == 0:
            # print("the no leaf number list", number_list)
            raise Exception("No Leaf Node?!")
        return leaf_string, leaf_node_index

    def get_bert_embeddings_in_batches(self, texts, batch_size):
        embeddings = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)  # 将模型移动到 GPU:0

        dataset = TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch_texts in dataloader:
            print("in batch")
            try:
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", truncation=True, padding=True
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
            except Exception as e:
                print(f"Error tokenizing batch: {batch_texts}")
                print(e)
                exit()

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            # 检查输出
            if outputs is not None and outputs.last_hidden_state is not None:
                print("Shape of last_hidden_state:", outputs.last_hidden_state.shape)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            else:
                print("outputs or last_hidden_state is None")
            embeddings.extend(batch_embeddings)

        return embeddings

    def get_node_string_pair(
        self, rst_results_store_path, output_file="node_embeddings.npz"
    ):
        """获取每个节点的字符串表示和对应的embeddings`

        Parameters
        ----------
        rst_results_store_path : str
            新的rst结果存储路径
        output_file : str, optional
            by default "node_embeddings.npz"

        Returns
        -------
        _ : dict
            key: node_string, value: embeddings
        """
        directory = os.path.dirname(output_file)

        # 查找目录中所有的 .npz 文件
        npz_files = glob.glob(os.path.join(directory, "*.npz"))
        if npz_files:
            print(f"Found specified .npz file: {npz_files}")
            return
        rst_results = self.get_stored_rst(rst_results_store_path)
        print("new rst_results length", len(rst_results))
        data_to_save = []

        premise_texts = []
        hypothesis_texts = []
        premise_indices = []
        hypothesis_indices = []

        for index, rst_result in enumerate(rst_results):
            print("index", index)  # debug check

            if rst_result["rst_relation_premise"] == [
                "NONE"
            ]:  # 对没有rst结构的句子进行处理
                print(
                    'type(rst_result["leaf_node_string_pre"]):',
                    type(rst_result["leaf_node_string_pre"][0][1]),
                )
                premise_texts.append(
                    str(rst_result["leaf_node_string_pre"][0][1])
                )  # [[node, string], [node, string]...]
                premise_indices.append([1])  # [[1]]
            else:
                pre_leaf_node_string_list = rst_result["leaf_node_string_pre"]
                pre_leaf_node_index, pre_leaf_string = zip(*pre_leaf_node_string_list)
                premise_texts.extend(pre_leaf_string)
                premise_indices.append(
                    pre_leaf_node_index
                )  # [[1, 2, 3, 4], [1, 2, 3, 4]...]

            if rst_result["rst_relation_hypothesis"] == ["NONE"]:
                hypothesis_texts.append(str(rst_result["leaf_node_string_hyp"][0][1]))
                hypothesis_indices.append([1])
            else:
                hyp_leaf_node_string_list = rst_result["leaf_node_string_hyp"]
                hyp_leaf_node_index, hyp_leaf_string = zip(*hyp_leaf_node_string_list)
                hypothesis_texts.extend(hyp_leaf_string)
                hypothesis_indices.append(hyp_leaf_node_index)

        for index, text in enumerate(premise_texts):
            if text == "":
                premise_texts[index] = "EMPTY"
                print("premise_texts", index)
        for index, text in enumerate(hypothesis_texts):
            if text == "":
                hypothesis_texts[index] = "EMPTY"
                print("hypothesis_texts", index)
        # 批量获取嵌入
        premise_embeddings = self.get_bert_embeddings_in_batches(
            premise_texts, batch_size=128
        )
        hypothesis_embeddings = self.get_bert_embeddings_in_batches(
            hypothesis_texts, batch_size=128
        )

        # 重新组织嵌入结果
        premise_offset = 0
        hypothesis_offset = 0

        for i, rst_result in enumerate(rst_results):
            print("in save", i)
            node_embeddings_premise = [
                (node, premise_embeddings[premise_offset + j])
                for j, node in enumerate(premise_indices[i])
            ]
            premise_offset += len(
                premise_indices[i]
            )  # 这里加offset的原因是，embedding是一整个列表存的

            node_embeddings_hypothesis = [
                (node, hypothesis_embeddings[hypothesis_offset + j])
                for j, node in enumerate(hypothesis_indices[i])
            ]
            hypothesis_offset += len(hypothesis_indices[i])

            data_to_save.append(
                {
                    "premise": node_embeddings_premise,
                    "hypothesis": node_embeddings_hypothesis,
                }
            )  # 如果是没有rst信息的，比如node_embeddings_premise就是：[(0, embedding)]
            if (i % 5000) == 0 and (i != 0):  # i=0的时候不保存
                filename = output_file + str(i) + ".npz"
                torch.save(data_to_save, filename)
                data_to_save = []
                print("5000 pairs saved")
        if data_to_save and i < 5000:
            filename = output_file
            torch.save(data_to_save, filename)
        elif data_to_save:
            filename = output_file + str(i) + ".npz"
            torch.save(data_to_save, filename)

        print("get all embeddings")

        # print("get all embeddings")
        # self.save_embeddings_in_chunks(data_to_save, output_file)
        # # torch.save(data_to_save, output_file)
        # print(f"Node embeddings saved to {output_file}")
        return data_to_save

    def get_hypothesis_emb(self, train_data, emb_output_path):
        """
        从模型输出中获取假设的embedding
        list中是三个为一组的tuple，每个tuple中包含三个embedding"""
        directory = os.path.dirname(emb_output_path)
        npz_files = glob.glob(os.path.join(directory, "*.npz"))
        if npz_files:
            # 如果找到 .npz 文件
            if emb_output_path in npz_files:
                # 如果指定的文件存在，直接读取它
                print(f"Found specified .npz file: {emb_output_path}")
                return
        hypothesis_list = []
        for item in train_data:
            hypotheis_dict = item["model_output"]
            entailment_hypothesis = hypotheis_dict["entail_hypothesis"]
            neutral_hypothesis = hypotheis_dict["neutral_hypothesis"]
            contradict_hypothesis = hypotheis_dict["conflict_hypothesis"]
            hypothesis_list.append(entailment_hypothesis)
            hypothesis_list.append(neutral_hypothesis)
            hypothesis_list.append(contradict_hypothesis)

        hypothesis_embeddings = self.get_bert_embeddings_in_batches(
            hypothesis_list, batch_size=128
        )
        # 创建一个保存元组的列表，每个元组包含3个 embedding
        grouped_embeddings = [
            tuple(hypothesis_embeddings[i : i + 3])
            for i in range(0, len(hypothesis_embeddings), 3)
        ]

        torch.save(grouped_embeddings, emb_output_path)

    def load_embeddings(self, file_path):
        data = torch.load(file_path)
        return data

    def rewrite_rst_result(self, rst_results_store_paths, new_rst_results_store_path):
        """重写rst结果，将每个节点的核心性和关系分别提取出来，便于构建dgl图

        Parameters
        ----------
        rst_results_store_path : str
            原来的rst结果存储路径
        new_rst_results_store_path : str
            新的rst结果存储路径
        """

        rst_results = self.get_stored_rst(rst_results_store_paths)
        new_rst_results = []
        for rst_result in rst_results:
            single_dict = {}
            rst_relation_premise = []
            rst_relation_hypothesis = []
            premise_node_nuclearity = [(0, "root")]
            hypothesis_node_nuclearity = [(0, "root")]

            if rst_result["pre_node_number"] == 1:
                premise_node_nuclearity.append((1, "single"))
                single_dict["premise_node_nuclearity"] = premise_node_nuclearity
                single_dict["rst_relation_premise"] = ["NONE"]
                single_dict["pre_node_type"] = [1, 0]
                single_dict["leaf_node_string_pre"] = [
                    [1, rst_result["pre_node_string"][0]]
                ]
            else:
                pre_leaf_string, pre_leaf_node_index = self.find_leaf_node(
                    rst_result["pre_node_number"], rst_result["pre_node_string"]
                )  # 记录叶子节点及其对应的字符串
                if len(pre_leaf_string) != len(pre_leaf_node_index):
                    raise ValueError(
                        "pre_leaf_string and pre_leaf_node_index must have the same length"
                    )
                combined_list_pre = list(zip(pre_leaf_node_index, pre_leaf_string))
                single_dict["leaf_node_string_pre"] = combined_list_pre
                pre_rel = rst_result["pre_node_relations"]
                pre_tree = rst_result["pre_tree"]
                for index, item in enumerate(
                    pre_rel
                ):  # 对premise中的每个关系组进行分析，分别提取左右两个子树的关系，以及子节点的核心性
                    rel_left = item["rel_left"]
                    src_left = pre_tree[index * 2][0] - 1  # dgl的节点从0开始，所以要减1
                    dst_left = pre_tree[index * 2][1] - 1
                    node_nuclearity = item[
                        "nuc_left"
                    ]  # 只取目标节点的核心性，这样不会重复
                    relation_1 = (src_left, dst_left, rel_left)
                    node_nuclearity_1 = (dst_left, node_nuclearity)
                    rst_relation_premise.append(relation_1)
                    premise_node_nuclearity.append(node_nuclearity_1)

                    rst_right = item["rel_right"]
                    src_right = pre_tree[index * 2 + 1][0] - 1
                    dst_right = pre_tree[index * 2 + 1][1] - 1
                    node_nuclearity = item["nuc_right"]
                    relation_2 = (src_right, dst_right, rst_right)
                    node_nuclearity_2 = (dst_right, node_nuclearity)
                    rst_relation_premise.append(relation_2)
                    premise_node_nuclearity.append(node_nuclearity_2)

                pre_child_node_list = [x - 1 for x in rst_result["pre_leaf_node"]]
                pre_node_type = [
                    0 if i in pre_child_node_list else 1
                    for i in range(len(pre_tree) + 1)
                ]
                single_dict["rst_relation_premise"] = rst_relation_premise
                single_dict["premise_node_nuclearity"] = premise_node_nuclearity
                single_dict["pre_node_type"] = pre_node_type

            # 接下来处理hypothesis
            if rst_result["hyp_node_number"] == 1:
                hypothesis_node_nuclearity.append((1, "single"))
                single_dict["hypothesis_node_nuclearity"] = hypothesis_node_nuclearity
                single_dict["rst_relation_hypothesis"] = ["NONE"]
                single_dict["hyp_node_type"] = [1, 0]
                single_dict["leaf_node_string_hyp"] = [
                    [1, rst_result["hyp_node_string"][0]]
                ]
            else:
                hyp_leaf_string, hyp_leaf_node_index = self.find_leaf_node(
                    rst_result["hyp_node_number"], rst_result["hyp_node_string"]
                )
                if len(hyp_leaf_string) != len(hyp_leaf_node_index):
                    raise ValueError(
                        "hyp_leaf_string and hyp_leaf_node_index must have the same length"
                    )
                combined_list_hyp = list(zip(hyp_leaf_node_index, hyp_leaf_string))
                single_dict["leaf_node_string_hyp"] = combined_list_hyp
                hyp_rel = rst_result["hyp_node_relations"]
                hyp_tree = rst_result["hyp_tree"]
                for index, item in enumerate(
                    hyp_rel
                ):  # 对hypothesis中的每个关系组进行分析，分别提取左右两个子树的关系，以及子节点的核心性
                    rel_left = item["rel_left"]
                    src_left = hyp_tree[index * 2][0] - 1  # dgl的节点从0开始，所以要减1
                    dst_left = hyp_tree[index * 2][1] - 1
                    node_nuclearity = item["nuc_left"]
                    relation_1 = (src_left, dst_left, rel_left)
                    node_nuclearity_1 = (dst_left, node_nuclearity)
                    rst_relation_hypothesis.append(relation_1)
                    hypothesis_node_nuclearity.append(node_nuclearity_1)

                    rst_right = item["rel_right"]
                    src_right = hyp_tree[index * 2 + 1][0] - 1
                    dst_right = hyp_tree[index * 2 + 1][1] - 1
                    node_nuclearity = item["nuc_right"]
                    relation_2 = (src_right, dst_right, rst_right)
                    node_nuclearity_2 = (dst_right, node_nuclearity)
                    rst_relation_hypothesis.append(relation_2)
                    hypothesis_node_nuclearity.append(node_nuclearity_2)

                hyp_child_node_list = [x - 1 for x in rst_result["hyp_leaf_node"]]
                hyp_node_type = [
                    0 if i in hyp_child_node_list else 1
                    for i in range(len(hyp_tree) + 1)
                ]
                single_dict["rst_relation_hypothesis"] = rst_relation_hypothesis
                single_dict["hypothesis_node_nuclearity"] = hypothesis_node_nuclearity
                single_dict["hyp_node_type"] = hyp_node_type

            new_rst_results.append(single_dict)

        self.write_jsonl(new_rst_results_store_path, new_rst_results)

    def find_lexical_chains(
        self, rst_results, node_features1, node_features2, threshold=0.8
    ):
        # node_feature {node_id: node_embedding, ...}
        """找到两文章之间的lexical chains"""
        # pre_leaf_node_index, pre_leaf_string = zip(*rst_results["leaf_node_string_pre"])
        # hyp_leaf_node_index, hyp_leaf_string = zip(*rst_results["leaf_node_string_hyp"])

        # print("pre_leaf_string:", pre_leaf_string)
        pre_length = len(rst_results["premise_node_nuclearity"])
        pre2_length = len(rst_results["hypothesis_node_nuclearity"])
        max_length = max(pre_length, pre2_length)

        chains_matrix = np.zeros(
            (pre_length, pre2_length)
        )  # 建图的时候还是建所有的节点的

        for node_id1, embedding1 in node_features1.items():
            # 确保embedding1是numpy数组并reshape为2D
            emb1 = np.array(embedding1)
            if emb1.ndim == 1:
                emb1 = emb1.reshape(1, -1)
            # 遍历第二个字典
            for node_id2, embedding2 in node_features2.items():
                # 确保embedding2是numpy数组并reshape为2D
                emb2 = np.array(embedding2)
                if emb2.ndim == 1:
                    emb2 = emb2.reshape(1, -1)
                # 计算余弦相似度
                similarity = cosine_similarity(emb1, emb2)[0][0]
                # 如果相似度大于阈值，添加到结果列表
                if similarity > threshold:
                    chains_matrix[node_id1][node_id2] = 1

        # 找出最大值和最小值
        amin, amax = chains_matrix.min(), chains_matrix.max()
        # 对数组进行归一化处理
        epsilon = 1e-7  # 举例epsilon为一个小的正值
        chains_matrix = (chains_matrix - amin) / (amax - amin + epsilon)
        # chains_matrix = (chains_matrix - amin) / (amax - amin)
        # print("chains_matrix", chains_matrix)
        return chains_matrix

    def save_lexical_matrix(self, path, matrixes):
        with open(path, "wb") as f:
            pickle.dump(matrixes, f)

    def load_lexical_matrix(self, filename):
        with open(filename, "rb") as f:
            matrixes = pickle.load(f)
        return matrixes

    def store_or_get_lexical_matrixes(self, rst_results_store_path, emb_path):
        lexical_matrixes_path = os.path.join(
            self.save_dir_lexical, self.lexical_matrix_path
        )
        print("lexical path", lexical_matrixes_path)
        if os.path.exists(lexical_matrixes_path):
            matrixes = self.load_lexical_matrix(lexical_matrixes_path)
            print(f"Matrix shape: {matrixes[0].shape}")
            non_zero_indices = np.nonzero(matrixes[0])
            print(f"Non-zero indices: {non_zero_indices}")
            print("load stored lexical matrix")
            return matrixes
        rst_results = self.get_stored_rst(rst_results_store_path)
        embeddings = self.load_embeddings(emb_path)
        matrixes = []
        """存储或者获取lexical chains matrix"""
        for index, rst_result in enumerate(rst_results):
            node_features1 = extract_node_features(embeddings, index, "premise")
            node_features2 = extract_node_features(embeddings, index, "hypothesis")
            matrix = self.find_lexical_chains(
                rst_result, node_features1, node_features2
            )
            matrixes.append(matrix)

        if self.save_or_not:
            os.makedirs(self.save_dir_lexical, exist_ok=True)
            print("lexical matrix saved ")
            self.save_lexical_matrix(lexical_matrixes_path, matrixes)


if __name__ == "__main__":
    # 调用示例
    model_path = r"/mnt/nlp/yuanmengying/models/xlm_roberta_large"

    overall_save_dir = r"/mnt/nlp/yuanmengying/ymy/data/cross_document"
    graph_infos_dir = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_infos"
    ################################################################################
    # train_data_path = r"/mnt/nlp/yuanmengying/nli_data_generate/sample_1_generated_hypothesis.json"  # 先拿十条数据测试
    train_data_path = (
        r"/mnt/nlp/yuanmengying/nli_data_generate/all_generated_hypothesis.json"
    )
    ori_train_data_path = (
        r"/mnt/nlp/yuanmengying/nli_data_generate/GlobeSumm_train.json"
    )
    model_hyp_emb_output_path = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/train/hyp/hypothesis_embeddings.npz"
    ################################################################################
    dev_data_path = (
        r"/mnt/nlp/yuanmengying/nli_data_generate/valid_all_generated_hypothesis.json"
    )
    ori_dev_data_path = r"/mnt/nlp/yuanmengying/nli_data_generate/GlobeSumm_valid.json"
    dev_model_hyp_emb_output_path = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/dev/hyp/hypothesis_embeddings.npz"

    ################################################################################
    test_data_path = (
        r"/mnt/nlp/yuanmengying/nli_data_generate/test_all_generated_hypothesis.json"
    )
    ori_test_data_path = r"/mnt/nlp/yuanmengying/nli_data_generate/GlobeSumm_test.json"
    test_model_hyp_emb_output_path = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/test/hyp/hypothesis_embeddings.npz"

    ################################################################################
    # data_porcessor_train = Data_Processor(True, overall_save_dir, "train")
    # train_data, ori_data, train_rst_result = load_all_data(
    #     data_porcessor_train, train_data_path, ori_train_data_path
    # )
    # print("original train data length", len(train_data))
    # embedder_train = RSTEmbedder(model_path, graph_infos_dir, "train", True)
    # train_rst_results_store_paths = glob.glob(
    #     os.path.join(os.path.join(overall_save_dir, "train"), "*.jsonl")
    # )
    # print(train_rst_results_store_paths, "train_rst_results_store_paths")
    # embedder_train.rewrite_rst_result(
    #     train_rst_results_store_paths,
    #     os.path.join(overall_save_dir, "train/train1", "new_rst_result.jsonl"),
    # )
    # train_node_string_pairs = embedder_train.get_node_string_pair(
    #     os.path.join(overall_save_dir, "train/train1", "new_rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "train/pre", "node_embeddings.npz"),
    # )

    # embedder_train.get_hypothesis_emb(
    #     train_data, model_hyp_emb_output_path
    # )  # 获取假设的embedding
    # train_matrix = embedder_train.store_or_get_lexical_matrixes(
    #     os.path.join(overall_save_dir, "train/train1", "new_rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "train/pre", "node_embeddings.npz"),
    # )

    ################################################################################

    # data_processor_dev = Data_Processor(True, overall_save_dir, "dev")
    # dev_data, ori_data, dev_rst_result = load_all_data(
    #     data_processor_dev, dev_data_path, ori_dev_data_path
    # )
    # print("original dev data length)", len(dev_data))
    # embedder_dev = RSTEmbedder(model_path, graph_infos_dir, "dev", True)
    # dev_rst_results_store_paths = glob.glob(
    #     os.path.join(os.path.join(overall_save_dir, "dev"), "*.jsonl")
    # )
    # # print(len(dev_rst_results_store_paths), "dev_rst_results_store_paths")
    # embedder_dev.rewrite_rst_result(
    #     dev_rst_results_store_paths,
    #     os.path.join(overall_save_dir, "dev/dev1", "new_rst_result.jsonl"),
    # )
    # dev_node_string_pairs = embedder_dev.get_node_string_pair(
    #     os.path.join(overall_save_dir, "dev/dev1", "new_rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "dev/pre", "node_embeddings.npz"),
    # )
    # embedder_dev.get_hypothesis_emb(
    #     dev_data, dev_model_hyp_emb_output_path
    # )
    # dev_matrix = embedder_dev.store_or_get_lexical_matrixes(
    #     os.path.join(overall_save_dir, "dev/dev1", "new_rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "dev/pre", "node_embeddings.npz"),
    # )
    ################################################################################

    data_processor_test = Data_Processor(True, overall_save_dir, "test")
    test_data, ori_data, test_rst_result = load_all_data(
        data_processor_test, test_data_path, ori_test_data_path
    )
    print("original test data length", len(test_data))

    embedder_test = RSTEmbedder(model_path, graph_infos_dir, "test", True)

    embedder_test.rewrite_rst_result(
        os.path.join(overall_save_dir, "test", "rst_result.jsonl"),
        os.path.join(overall_save_dir, "test/test1", "new_rst_result.jsonl"),
    )

    embedder_test.get_hypothesis_emb(
        test_data, test_model_hyp_emb_output_path
    )  # 获取假设的embedding

    test_node_string_pairs = embedder_test.get_node_string_pair(
        os.path.join(overall_save_dir, "test/test1", "new_rst_result.jsonl"),
        os.path.join(overall_save_dir, "test/pre", "node_embeddings.npz"),
    )

    test_matrix = embedder_test.store_or_get_lexical_matrixes(
        os.path.join(overall_save_dir, "test/test1", "new_rst_result.jsonl"),
        os.path.join(overall_save_dir, "test/pre", "node_embeddings.npz"),
    )
