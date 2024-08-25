import json
import torch
import numpy as np
import nltk
import os
import pickle
import nltk
from nltk.tokenize import word_tokenize

# 下载punkt分词器模型（如果还没有下载过）
# nltk.download('punkt')
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.DM_RST import *


def load_all_data(data_processor, train_data_path):
    train_data = data_processor.load_json(train_data_path)
    train_data = train_data[151500:155000]
    rst_results = data_processor.get_rst(train_data)
    return train_data, rst_results


# 主要是获取处理后的数据，包括rst树的信息，以及节点的字符串表示和bert embeddings，以及词汇链信息
class Data_Processor:
    def __init__(self, mode, save_dir, purpose):
        self.save_dir = os.path.join(save_dir, purpose)
        self.rst_path = "rst_result.jsonl"
        self.save_or_not = mode

    def load_json(self, json_path):
        with open(json_path, "r") as f:
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

    def get_rst(self, data):
        """
        获取前提和假设的rst🌲 分析结果，包括树节点、节点的string、节点的核性、节点间的关系
        :param data:
        :return:
        """
        rst_results_store_path = os.path.join(self.save_dir, self.rst_path)
        print("the rst result path", rst_results_store_path)
        if os.path.exists(rst_results_store_path):
            rst_results = self.get_stored_rst(rst_results_store_path)
            return rst_results

        my_rst_tree = RST_Tree()
        model = my_rst_tree.init_model()  # 初始化模型
        precess_rst_tree = precess_rst_result()
        rst_results = []
        count = 151500

        for index, i in enumerate(data):
            if index == 285:
                print("the 151785 data", i)
                count += 1
                continue
            input_sentences, all_segmentation_pred, all_tree_parsing_pred = (
                my_rst_tree.inference(model, [i["premise"], i["hypothesis"]])
            )

            segments_pre = precess_rst_tree.merge_strings(
                input_sentences[0], all_segmentation_pred[0]
            )  # 获取单个edu的string
            segments_hyp = precess_rst_tree.merge_strings(
                input_sentences[1], all_segmentation_pred[1]
            )  # 获取单个edu的string
            # 提取出rst结构，字符串形式
            if all_tree_parsing_pred[0][0] == "NONE":
                node_number_pre = 1
                node_string_pre = [segments_pre]
                RelationAndNucleus_pre = "NONE"
                tree_pre = [[1, 1]]
                leaf_node_pre = [1]
                parent_dict_pre = {"1_1": 1}
            else:
                rst_info_pre = all_tree_parsing_pred[0][0].split()
                node_number_pre, node_string_pre = precess_rst_tree.use_rst_info(
                    rst_info_pre, segments_pre
                )  # 遍历RST信息，提取关系和标签信息
                RelationAndNucleus_pre = precess_rst_tree.get_RelationAndNucleus(
                    rst_info_pre
                )  # 提取核性和关系
                parent_dict_pre, leaf_node_pre, tree_pre = self.get_tree(
                    node_number_pre
                )
            if all_tree_parsing_pred[1][0] == "NONE":
                node_number_hyp = 1
                node_string_hyp = [segments_hyp]
                RelationAndNucleus_hyp = "NONE"
                tree_hyp = [[1, 1]]
                leaf_node_hyp = [1]
                parent_dict_hyp = {"1_1": 1}
            else:
                rst_info_hyp = all_tree_parsing_pred[1][
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
            if i["label"] == "entailment":
                numbered_label = 0
            elif i["label"] == "not_entailment":
                numbered_label = 1
            else:
                numbered_label = None
                print("label is not in scale, please check", i["label"])
            rst_results.append(
                {
                    "pre_node_number": node_number_pre,
                    "pre_node_string": node_string_pre,
                    "pre_node_relations": RelationAndNucleus_pre,
                    "pre_tree": tree_pre,
                    "pre_leaf_node": leaf_node_pre,
                    "pre_parent_dict": parent_dict_pre,
                    "hyp_node_number": node_number_hyp,
                    "hyp_node_string": node_string_hyp,
                    "hyp_node_relations": RelationAndNucleus_hyp,
                    "hyp_tree": tree_hyp,
                    "hyp_leaf_node": leaf_node_hyp,
                    "hyp_parent_dict": parent_dict_hyp,
                    "label": numbered_label,
                }
            )
            print(count, "count")
            count += 1  # 增加计数器
            # 每5000条保存一次
            if count % 500 == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                rst_name = str(count) + "_rst_result.jsonl"
                self.write_jsonl(os.path.join(self.save_dir, rst_name), rst_results)
                print(
                    f"Saved {len(rst_results)} records to {os.path.join(self.save_dir, self.rst_path)}"
                )
                rst_results = []  # 清空列表以便下次使用
        # 循环结束后，保存剩余的结果
        if rst_results:
            os.makedirs(self.save_dir, exist_ok=True)
            rst_name = "left_rst_result.jsonl"
            self.write_jsonl(os.path.join(self.save_dir, rst_name), rst_results)
            print(
                f"Saved {len(rst_results)} records to {os.path.join(self.save_dir, self.rst_path)}"
            )
        else:
            print("No left records to save.")
            # "pre_node_number":[[1,12,13,14]...]
            # "pre_node_string":[["edu1-12","edu13-14"]...]
            # "pre_node_relations":[{'nuc_left': 'Nucleus', 'nuc_right': 'Satellite', 'rel_left': 'span', 'rel_right': 'Topic-Comment'}...]
            # "pre_tree":[[1,2],[1,3],[2,4]...]
            # "pre_leaf_node":[8,12,13,14...]
            # "pre_parent_dict":{"1_12":2, "13_14":3,""...}
            # "label":0 or 1
            # 如果没有rst结构，那么直接跳过，但是

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


class RSTEmbedder:
    def __init__(self, model_path, save_dir, purpose, save_or_not):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(model_path)
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

    def get_stored_rst(self, path):
        rst_results = []
        with open(path, "r") as file:
            for line in file:
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
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

    def get_bert_embeddings(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def get_node_string_pair(
        self, rst_results_store_path, output_file="node_embeddings.npz"
    ):
        """获取每个节点的字符串表示和对应的bert embeddings`

        Parameters
        ----------
        rst_results_store_path : str
            新的rst结果存储路径
        output_file : str, optional
            by default "node_embeddings.npz"

        Returns
        -------
        _ : dict
            key: node_string, value: bert embeddings
        """
        rst_results = self.get_stored_rst(rst_results_store_path)
        data_to_save = []

        for rst_result in rst_results:
            if rst_result["rst_relation_premise"] == ["NONE"]:
                embeddings_premise = self.get_bert_embeddings(
                    rst_result["leaf_node_string_pre"][0][1]
                )  # debug check
                node_embeddings_premise = [(1, embeddings_premise)]
            else:
                pre_leaf_node_string_list = rst_result["leaf_node_string_pre"]
                # print(type(pre_leaf_node_string_list))
                pre_leaf_node_index, pre_leaf_string = zip(*pre_leaf_node_string_list)
                embeddings_premise = self.get_bert_embeddings(pre_leaf_string)
                node_embeddings_premise = [
                    (node, embedding)  # 存的时候都是从0开始的了
                    for node, embedding in zip(pre_leaf_node_index, embeddings_premise)
                ]
            if rst_result["rst_relation_hypothesis"] == ["NONE"]:
                embeddings_hypothesis = self.get_bert_embeddings(
                    rst_result["leaf_node_string_hyp"][0][1]
                )
                node_embeddings_hypothesis = [(1, embeddings_hypothesis)]
            else:
                hyp_leaf_node_string_list = rst_result["leaf_node_string_hyp"]
                hyp_leaf_node_index, hyp_leaf_string = zip(*hyp_leaf_node_string_list)
                embeddings_hypothesis = self.get_bert_embeddings(hyp_leaf_string)
                node_embeddings_hypothesis = [
                    (node, embedding)
                    for node, embedding in zip(
                        hyp_leaf_node_index, embeddings_hypothesis
                    )
                ]

            data_to_save.append(
                {
                    "premise": node_embeddings_premise,  # 这里的node_embeddings_premise是一个list，每个元素是一个tuple：（node,embedding)
                    "hypothesis": node_embeddings_hypothesis,  # 这里的node_embeddings_hypothesis是一个list，每个元素是一个tuple
                }
            )

        torch.save(data_to_save, output_file)
        print(f"Node embeddings saved to {output_file}")
        return data_to_save

    def load_embeddings(self, file_path):
        data = torch.load(file_path)
        return data

    def rewrite_rst_result(self, rst_results_store_path, new_rst_results_store_path):
        """重写rst结果，将每个节点的核心性和关系分别提取出来，便于构建dgl图

        Parameters
        ----------
        rst_results_store_path : str
            原来的rst结果存储路径
        new_rst_results_store_path : str
            新的rst结果存储路径
        """

        rst_results = self.get_stored_rst(rst_results_store_path)
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

            single_dict["label"] = str(rst_result["label"])

            new_rst_results.append(single_dict)

        self.write_jsonl(new_rst_results_store_path, new_rst_results)

    def find_lexical_chains(self, rst_results, threshold=0.8):
        """找到两段落之间的lexical chains"""
        pre_leaf_node_index, pre_leaf_string = zip(*rst_results["leaf_node_string_pre"])
        hyp_leaf_node_index, hyp_leaf_string = zip(*rst_results["leaf_node_string_hyp"])
        # print("pre_leaf_string:", pre_leaf_string)
        pre_length = len(rst_results["premise_node_nuclearity"])
        hyp_length = len(rst_results["hypothesis_node_nuclearity"])
        max_length = max(pre_length, hyp_length)
        # print("max_length", max_length)
        # print(pre_length, hyp_length, "pre_length, hyp_length")
        # print(
        #     pre_leaf_node_index,
        #     hyp_leaf_node_index,
        #     "pre_leaf_node_index, hyp_leaf_node_index",
        # )
        chains_matrix = np.zeros(
            (pre_length, hyp_length)
        )  # 建图的时候还是建所有的节点的

        # 遍历两个列表
        for i, text_a in enumerate(pre_leaf_string):
            words_a = set(word_tokenize(text_a.lower()))  # 分词并转换为小写集合
            for j, text_b in enumerate(hyp_leaf_string):
                words_b = set(word_tokenize(text_b.lower()))  # 分词并转换为小写集合
                # 检查是否有相同词汇
                if words_a & words_b:
                    chains_matrix[pre_leaf_node_index[i]][hyp_leaf_node_index[j]] = 1

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

    def store_or_get_lexical_matrixes(self, rst_results_store_path):
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
        matrixes = []
        """存储或者获取lexical chains matrix"""
        for rst_result in rst_results:
            matrix = self.find_lexical_chains(rst_result)
            matrixes.append(matrix)

        if self.save_or_not:
            os.makedirs(self.save_dir_lexical, exist_ok=True)
            print("lexical matrix saved ")
            self.save_lexical_matrix(lexical_matrixes_path, matrixes)


if __name__ == "__main__":
    # 调用示例
    model_path = r"/mnt/nlp/yuanmengying/models/roberta-large"
    # rst_results_store_path = r"/mnt/nlp/yuanmengying/ymy/data/temp_data_dev/rst_result.jsonl"  # 9988条训练数据
    # new_rst_results_store_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_dev/new_rst_result.jsonl"
    # )
    # embeeding_store_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_dev/node_embeddings.npz"
    # )
    overall_save_dir = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin"
    graph_infos_dir = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos"

    # train_data_path = r"/mnt/nlp/yuanmengying/ymy/data/DocNLI_dataset/new_train.json"
    # dev_data_path = r"/mnt/nlp/yuanmengying/ymy/data/DocNLI_dataset/new_dev.json"
    # test_data_path = r"/mnt/nlp/yuanmengying/ymy/data/DocNLI_dataset/new_test.json"
    train_data_path = r"/mnt/nlp/yuanmengying/ymy/data/DocNLI_dataset/train.json"
    dev_data_path = r"/mnt/nlp/yuanmengying/ymy/data/DocNLI_dataset/dev.json"
    test_data_path = r"/mnt/nlp/yuanmengying/ymy/data/DocNLI_dataset/test.json"

    # data_porcessor_train = Data_Processor(True, overall_save_dir, "train")
    data_processor_dev = Data_Processor(True, overall_save_dir, "dev")
    # data_processor_test = Data_Processor(True, overall_save_dir, "test")

    # train_data, train_rst_result = load_all_data(data_porcessor_train, train_data_path)
    # print("original train data length", len(train_data))
    dev_data, dev_rst_result = load_all_data(data_processor_dev, dev_data_path)
    print("original dev data length)", len(dev_data))
    # test_data, test_rst_result = load_all_data(data_processor_test, test_data_path)
    # print("original test data length", len(test_data))

    # embedder_train = RSTEmbedder(model_path, graph_infos_dir, "train", True)
    embedder_dev = RSTEmbedder(model_path, graph_infos_dir, "dev", True)
    # embedder_test = RSTEmbedder(model_path, graph_infos_dir, "test", True)

    # embedder_train.rewrite_rst_result(
    #     os.path.join(overall_save_dir, "train", "rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "train", "new_rst_result.jsonl"),
    # )
    # embedder_dev.rewrite_rst_result(
    #     os.path.join(overall_save_dir, "dev", "rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "dev", "new_rst_result.jsonl"),
    # )
    # embedder_test.rewrite_rst_result(
    #     os.path.join(overall_save_dir, "test", "rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "test", "new_rst_result.jsonl"),
    # )

    # # train_node_string_pairs = embedder_train.get_node_string_pair(
    # #     os.path.join(overall_save_dir, "train", "new_rst_result.jsonl"),
    # #     os.path.join(overall_save_dir, "train", "node_embeddings.npz"),
    # # )
    # dev_node_string_pairs = embedder_dev.get_node_string_pair(
    #     os.path.join(overall_save_dir, "dev", "new_rst_result.jsonl"),
    #     os.path.join(overall_save_dir, "dev", "node_embeddings.npz"),
    # )
    # # test_node_string_pairs = embedder_test.get_node_string_pair(
    # #     os.path.join(overall_save_dir, "test", "new_rst_result.jsonl"),
    # #     os.path.join(overall_save_dir, "test", "node_embeddings.npz"),
    # # )

    # # train_matrix = embedder_train.store_or_get_lexical_matrixes(
    # #     os.path.join(overall_save_dir, "train", "new_rst_result.jsonl")
    # # )
    # dev_matrix = embedder_dev.store_or_get_lexical_matrixes(
    #     os.path.join(overall_save_dir, "dev", "new_rst_result.jsonl")
    # )
    # # test_matrix = embedder_test.store_or_get_lexical_matrixes(
    # #     os.path.join(overall_save_dir, "test", "new_rst_result.jsonl")
    # # )

    # # embedder.rewrite_rst_result(rst_results_store_path, new_rst_results_store_path)
    # # node_string_pairs = embedder.get_node_string_pair(
    # #     new_rst_results_store_path, embeeding_store_path
    # # )
    # # matrix = embedder.store_or_get_lexical_matrixes(new_rst_results_store_path)
