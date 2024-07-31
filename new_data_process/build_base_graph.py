import dgl
import torch
import torch.nn.functional as F
import networkx as nx
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn

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


def merge_graphs(g_premise, g_hypothesis, lexical_chain):
    g_premise = g_premise.to(torch.device("cpu"))
    g_hypothesis = g_hypothesis.to(torch.device("cpu"))

    num_nodes_premise = g_premise.num_nodes()
    num_nodes_hypothesis = g_hypothesis.num_nodes()

    # 获取所有可能的边类型
    all_edge_types = list(set(g_premise.etypes).union(set(g_hypothesis.etypes)))

    # 初始化合并图的数据结构
    combined_graph_data = {
        ("node", etype, "node"): ([], []) for etype in all_edge_types
    }

    # 添加 g_premise 的边
    for etype in g_premise.etypes:
        src, dst = g_premise.edges(etype=etype)
        combined_graph_data[("node", etype, "node")][0].extend(src.tolist())
        combined_graph_data[("node", etype, "node")][1].extend(dst.tolist())

    # 添加 g_hypothesis 的边，并调整索引
    for etype in g_hypothesis.etypes:
        src, dst = g_hypothesis.edges(etype=etype)
        combined_graph_data[("node", etype, "node")][0].extend(
            (src + num_nodes_premise).tolist()
        )
        combined_graph_data[("node", etype, "node")][1].extend(
            (dst + num_nodes_premise).tolist()
        )

    # 添加 lexical_chain 的边，假设边的类型是 "lexical"
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

    # 确保所有边类型在所有图中都存在
    for etype in all_edge_types:
        if ("node", etype, "node") not in combined_graph_data:
            combined_graph_data[("node", etype, "node")] = ([], [])

    # 创建合并后的图
    num_combined_nodes = num_nodes_premise + num_nodes_hypothesis
    g_combined = dgl.heterograph(
        combined_graph_data, num_nodes_dict={"node": num_combined_nodes}
    )

    # 复制节点特征
    combined_features = torch.zeros(
        (num_combined_nodes, g_premise.ndata["feat"].shape[1]), dtype=torch.float32
    )
    combined_features[:num_nodes_premise] = g_premise.ndata["feat"].clone().detach()
    combined_features[num_nodes_premise:] = g_hypothesis.ndata["feat"].clone().detach()
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
        # inputs是节点的特征
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = {"node": g.ndata["feat"]}
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata["h"] = h["node"]
            # 通过平均读出值来计算单图的表征
            hg = dgl.mean_nodes(g, "h")
            return self.classify(hg)
