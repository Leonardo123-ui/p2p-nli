import dgl
import torch
import torch.nn.functional as F
import networkx as nx
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv

relation_types = [
    "Temporal",
    "TextualOrganization",
    "Joint",
    "Topic-Comment",
    "Comparison",
    "Condition",
    "Contrast",
    "Evaluation",
    "Topic-Change",
    "Summary",
    "Manner-Means",
    "Attribution",
    "Cause",
    "Background",
    "Enablement",
    "Explanation",
    "Same-Unit",
    "Elaboration",
    "span",  # 可以考虑去掉，因为关系不密切
    "lexical",  # 词汇链
]


def build_graph(node_features, node_types, rst_relations):
    """
    创建DGL图并添加节点及其特征。
    - node_features: 字典，键为节点ID，值为节点嵌入特征。
    - node_types: 列表，表示节点类型（0表示父节点，1表示子节点）。
    - rst_relations: 列表，包含所有的RST关系，形式为 (parent_node, child_node, relation_type)。
    """
    num_nodes = len(node_types)

    # 创建从父节点到子节点及关系类型的映射
    parent_to_children = {}
    if rst_relations == ["NONE"]:
        rst_relations = [[0, 1, "span"]]
    # print("rst_relations:", rst_relations)
    # for rst_relation in rst_relations:
    #     print("rst_relation:", rst_relation)
    for parent, child, rel_type in rst_relations:
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append((child, rel_type))

    # 初始化图数据结构
    graph_data = {("node", rel_type, "node"): ([], []) for rel_type in relation_types}

    for parent, children in parent_to_children.items():
        for child, rel_type in children:
            graph_data[("node", rel_type, "node")][0].append(parent)
            graph_data[("node", rel_type, "node")][1].append(child)
            # assuming undirected graph, add reverse edge
            graph_data[("node", rel_type, "node")][0].append(child)
            graph_data[("node", rel_type, "node")][1].append(parent)

    graph = dgl.heterograph(graph_data, num_nodes_dict={"node": num_nodes})

    # print(f"Number of nodes in the graph: {graph.number_of_nodes()}")

    # 初始化特征矩阵，确保所有节点都有特征
    feature_dim = len(next(iter(node_features.values())))
    features = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    for node_id, embedding in node_features.items():
        features[node_id] = (
            torch.tensor(embedding, dtype=torch.float32).clone().detach()
        )

    # 使用 networkx 进行拓扑排序，确定节点的层级顺序
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from([(parent, child) for parent, child, _ in rst_relations])
    topo_order = list(nx.topological_sort(nx_graph))

    # 从树的底部开始填充父节点特征
    for node in reversed(topo_order):
        if node_types[node] == 1:  # 父节点
            if node in parent_to_children:
                child_embeddings = [
                    features[child] for child, _ in parent_to_children[node]
                ]
                if child_embeddings:
                    parent_feature = sum(child_embeddings) / len(child_embeddings)
                    features[node] = parent_feature

    graph.ndata["feat"] = features.clone().detach()

    return graph


def merge_graphs(g_premise, g_premise2, lexical_chain, rel_names_short):
    num_nodes_premise = g_premise.num_nodes()
    num_nodes_hypothesis = g_premise2.num_nodes()

    # 获取所有可能的边类型，并过滤出关注的边类型
    all_edge_types = list(set(g_premise.etypes).union(set(g_premise2.etypes)))
    focused_edge_types = [etype for etype in all_edge_types if etype in rel_names_short]

    # 初始化合并图的数据结构
    combined_graph_data = {
        ("node", etype, "node"): ([], []) for etype in focused_edge_types
    }

    # 添加 g_premise 的边
    for etype in g_premise.etypes:
        if etype in rel_names_short:  # 只处理关注的边类型
            src, dst = g_premise.edges(etype=etype)
            combined_graph_data[("node", etype, "node")][0].extend(src.tolist())
            combined_graph_data[("node", etype, "node")][1].extend(dst.tolist())

    # 添加 g_premise2 的边，并调整索引
    for etype in g_premise2.etypes:
        if etype in rel_names_short:  # 只处理关注的边类型
            src, dst = g_premise2.edges(etype=etype)
            combined_graph_data[("node", etype, "node")][0].extend(
                (src + num_nodes_premise).tolist()
            )
            combined_graph_data[("node", etype, "node")][1].extend(
                (dst + num_nodes_premise).tolist()
            )

    # 添加 lexical_chain 的边，假设边的类型是 "lexical"
    if "lexical" in rel_names_short:  # 只处理关注的边类型
        src_nodes, dst_nodes = [], []
        for i in range(num_nodes_premise):
            for j in range(num_nodes_hypothesis):
                if lexical_chain[i][j] > 0:
                    src_nodes.append(i)
                    dst_nodes.append(
                        j + num_nodes_premise
                    )  # Offset by number of nodes in premise

        if src_nodes:
            edge_type = "lexical"
            if ("node", edge_type, "node") not in combined_graph_data:
                combined_graph_data[("node", edge_type, "node")] = ([], [])
            combined_graph_data[("node", edge_type, "node")][0].extend(src_nodes)
            combined_graph_data[("node", edge_type, "node")][1].extend(dst_nodes)
            combined_graph_data[("node", edge_type, "node")][0].extend(dst_nodes)
            combined_graph_data[("node", edge_type, "node")][1].extend(src_nodes)

    # 创建合并后的图
    num_combined_nodes = num_nodes_premise + num_nodes_hypothesis
    g_combined = dgl.heterograph(
        combined_graph_data, num_nodes_dict={"node": num_combined_nodes}
    )

    # 复制节点特征
    combined_features = torch.zeros(
        (num_combined_nodes, g_premise.ndata["feat"].shape[1]),
        dtype=torch.float32,
        device=g_premise.device,
    )
    combined_features[:num_nodes_premise] = g_premise.ndata["feat"].clone().detach()
    combined_features[num_nodes_premise:] = g_premise2.ndata["feat"].clone().detach()
    g_combined.ndata["feat"] = combined_features

    return g_combined


# R-GAT模型
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names},
            aggregate="sum",
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names},
            aggregate="sum",
        )

    def forward(self, graph, inputs):

        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names, rel_names_short):
        super().__init__()
        self.rgcn_premise = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.rgcn_hypothesis = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.rgcn_combined = RGCN(
            in_dim, hidden_dim, hidden_dim, rel_names_short
        )  # 合并图只聚合重要的rst关系
        self.pooling = dglnn.AvgPooling()
        # 分类层，输出 n_classes（3 类）
        self.classify = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, combined_graphs):

        h_combined = self.rgcn_combined(
            combined_graphs, {"node": combined_graphs.ndata["feat"]}
        )
        hg_combined = self.pooling(combined_graphs, h_combined["node"])  # batch * 256

        return hg_combined

    def classify_hypothesis(self, hg_combined, hyp_emb):
        """
        验证和预测时使用：将前提和假设的嵌入拼接后进行分类。
        - hg_combined: 大前提的嵌入。
        - hyp_emb: 假设的嵌入。
        """
        # 将大前提嵌入与假设嵌入拼接
        hg_concat = torch.cat([hg_combined, hyp_emb], dim=1)  # 拼接前提和假设的嵌入
        logits = self.classify(hg_concat)  # 分类
        return logits
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)  # 返回类别 0:蕴含, 1:中立, 2:矛盾
