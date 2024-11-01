import dgl
import torch
import torch.nn.functional as F
import networkx as nx
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5ForConditionalGeneration

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


def merge_graphs(g_premise, g_hypothesis, lexical_chain, rel_names_short):
    num_nodes_premise = g_premise.num_nodes()
    num_nodes_hypothesis = g_hypothesis.num_nodes()

    # 获取所有可能的边类型，并过滤出关注的边类型
    all_edge_types = list(set(g_premise.etypes).union(set(g_hypothesis.etypes)))
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

    # 添加 g_hypothesis 的边，并调整索引
    for etype in g_hypothesis.etypes:
        if etype in rel_names_short:  # 只处理关注的边类型
            src, dst = g_hypothesis.edges(etype=etype)
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
    combined_features[num_nodes_premise:] = g_hypothesis.ndata["feat"].clone().detach()
    g_combined.ndata["feat"] = combined_features

    return g_combined


class RGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names):
        super().__init__()

        # 第一层异构图卷积
        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    num_heads=4,  # 注意力头数
                    feat_drop=0.1,
                    attn_drop=0.1,
                    residual=True,
                )
                for rel in rel_names
            },
            aggregate="mean",
        )

        # 第二层异构图卷积
        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(
                    in_feats=hidden_dim * 4,  # 因为第一层有4个头，所以输入维度要乘4
                    out_feats=out_dim,
                    num_heads=1,  # 最后一层通常使用1个头
                    feat_drop=0.1,
                    attn_drop=0.1,
                    residual=True,
                )
                for rel in rel_names
            },
            aggregate="mean",
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, g, inputs):
        """
        g: DGLHeteroGraph
        inputs: 节点特征字典 {"node_type": features}
        """
        # 第一层卷积
        h = self.conv1(g, inputs)
        h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

        # 第二层卷积
        h = self.conv2(g, h)
        out = {k: v.squeeze(1) for k, v in h.items()}

        return out


def check_input_distribution(encoder_hidden_states):
    print(f"Input shape: {encoder_hidden_states.shape}")
    print(f"Mean: {encoder_hidden_states.mean().item():.4f}")
    print(f"Std: {encoder_hidden_states.std().item():.4f}")
    print(f"Max: {encoder_hidden_states.max().item():.4f}")
    print(f"Min: {encoder_hidden_states.min().item():.4f}")
    if encoder_hidden_states.requires_grad:
        print("Has gradient")
        if encoder_hidden_states.grad is not None:
            print(f"Gradient mean: {encoder_hidden_states.grad.mean().item():.4f}")
            print(f"Gradient std: {encoder_hidden_states.grad.std().item():.4f}")


class InputNormalization(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.scaling_factor = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 对输入进行归一化，并添加可学习的缩放因子
        normalized = self.layer_norm(x)
        return normalized * self.scaling_factor


class ExplainableHeteroClassifier(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_classes,
        rel_names,
        # decoder_name="/mnt/nlp/yuanmengying/models/bart-large",
        # decoder_name="/mnt/nlp/yuanmengying/models/flan-t5-large",
        decoder_name="/mnt/nlp/yuanmengying/models/t5-small",
    ):
        super().__init__()
        self.n_classes = n_classes

        # 图编码器
        self.rgat = RGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=rel_names,
        )

        self.pooling = dglnn.AvgPooling()

        # 分类器
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

        # 加载BART模型和分词器
        # self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
        # self.bart = BartForConditionalGeneration.from_pretrained(decoder_name)
        # 加载T5模型和分词器
        self.tokenizer = T5Tokenizer.from_pretrained(decoder_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(decoder_name)
        self.t5.config.use_cache = False  # 禁用KV缓存
        self.t5.gradient_checkpointing_enable()  # 启用梯度检查点
        self.feature_proj = nn.Sequential(
            nn.Linear(
                hidden_dim * 2, self.t5.config.d_model
            ),  # T5的隐藏层大小为d_model
            nn.GELU(),
            nn.LayerNorm(self.t5.config.d_model),
            nn.Dropout(0.1),
        )
        # 输出层：将BART隐藏状态映射到词表
        # self.output_proj = nn.Linear(self.bart.config.hidden_size, len(self.tokenizer))
        self.output_proj = nn.Linear(self.t5.config.d_model, len(self.tokenizer))
        self.input_norm = InputNormalization(hidden_size=1024)

    def encode_graph_combined(self, graph):
        """编码输入图"""
        h = self.rgat(graph, {"node": graph.ndata["feat"]})
        graph_repr = self.pooling(graph, h["node"])
        return graph_repr

    def forward(self, combined_graph_repr, hyp_emb, explanation=None):
        # 1. 合并特征

        combined_features = torch.cat([combined_graph_repr, hyp_emb], dim=1)

        # 投影到T5期望的隐藏层维度
        decoder_inputs_embeds = self.feature_proj(combined_features)

        # 2. 分类
        logits = self.classifier(combined_features)
        outputs = {"logits": logits}

        # 3. 生成解释
        if explanation is not None:
            # 将目标文本转化为模型输入
            explanation_inputs = self.tokenizer(
                explanation,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                return_attention_mask=True,  # 确保获取attention mask
            ).to(decoder_inputs_embeds.device)
            target_length = explanation_inputs["input_ids"].size(1)  # 这里是 110

            # 更安全的特征扩展方式
            encoder_hidden_states = decoder_inputs_embeds.unsqueeze(
                1
            )  # [batch_size, 1, hidden_size]
            encoder_hidden_states = encoder_hidden_states.repeat(
                1, target_length, 1
            )  # [batch_size, seq_len, hidden_size]
            # 添加归一化，防止数值不稳定
            encoder_hidden_states = self.input_norm(encoder_hidden_states)
            # 检查并处理数值问题
            if torch.isnan(encoder_hidden_states).any():
                print("Warning: NaN in encoder_hidden_states")
                # 可以添加一些处理逻辑
                encoder_hidden_states = torch.nan_to_num(encoder_hidden_states)
            # 添加attention mask
            attention_mask = explanation_inputs["attention_mask"]
            # 将编码后的特征作为输入，将标准输出作为labels引导生成，得到主要的生成loss
            explanation_outputs = self.t5(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoder_hidden_states
                ),
                attention_mask=attention_mask,  # 添加attention mask
                labels=explanation_inputs["input_ids"],
                return_dict=True,
            )
            if (
                hasattr(explanation_outputs, "loss")
                and explanation_outputs.loss is not None
            ):
                gen_loss = explanation_outputs.loss
                if torch.isnan(gen_loss).any():
                    print("Warning: NaN in generation loss")
                    gen_loss = torch.zeros_like(gen_loss)

                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                outputs["gen_loss"] = gen_loss
                outputs["explanation"] = explanation_outputs.logits

            # 计算序列内的diversity
            cosine_sim = torch.cosine_similarity(
                encoder_hidden_states.unsqueeze(
                    2
                ),  # [batch_size, seq_len, 1, hidden_size]
                encoder_hidden_states.unsqueeze(
                    1
                ),  # [batch_size, 1, seq_len, hidden_size]
                dim=-1,
            )  # [batch_size, seq_len, seq_len]

            # 移除对角线（自身的相似度）
            mask = torch.eye(cosine_sim.size(1), device=cosine_sim.device)
            cosine_sim = cosine_sim * (1 - mask)

            # diversity loss: 最小化序列内的相似度
            diversity_loss = cosine_sim.mean()
            # 组合损失
            total_loss = explanation_outputs.loss + 0.1 * diversity_loss

            # 获取生成的logits
            outputs["explanation"] = explanation_outputs.logits
            gen_loss = explanation_outputs.loss

            outputs["gen_loss"] = total_loss

        return outputs

    @torch.no_grad()
    def generate_explanation(
        self, combined_graph_repr, hyp_emb, max_length=128, num_beams=5
    ):
        features = torch.cat([combined_graph_repr, hyp_emb], dim=1)
        """使用Beam Search生成解释"""
        encoder_outputs = BaseModelOutput(
            last_hidden_state=self.feature_proj(features).unsqueeze(1)
        )

        # 使用T5的generate方法生成解释
        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,  # 防止重复
            length_penalty=1.0,  # 长度惩罚
            bad_words_ids=[[self.tokenizer.pad_token_id]],  # 避免生成 padding
            forced_bos_token_id=self.tokenizer.bos_token_id,  # 确保以 BOS 开始
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

        # 将生成的id转换为文本
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return generated_texts

    # def compute_loss(self, outputs, target_label):

    #         classification_loss = F.cross_entropy(outputs["logits"], target_label)
    #         gen_loss = F.cross_entropy(
    #             outputs["explanation"].view(-1, outputs["explanation"].size(-1)),
    #             outputs["explanation_inputs"]["input_ids"].view(-1),
    #             ignore_index=self.tokenizer.pad_token_id,
    #             label_smoothing=0.1,  # 可选：使用标签平滑
    #         )
    #         return classification_loss, gc

    # @torch.no_grad()
    # def generate_explanation(self, features, max_length=128):
    #     """自回归生成解释"""
    #     batch_size = features.size(0)

    #     # 准备decoder输入
    #     decoder_inputs = self.feature_proj(features)
    #     decoder_inputs = decoder_inputs.unsqueeze(1)

    #     # 初始化生成序列
    #     generated = (
    #         torch.ones(batch_size, 1).long().to(features.device)
    #         * self.tokenizer.bos_token_id
    #     )

    #     for _ in range(max_length):
    #         # decoder前向传播
    #         decoder_outputs = self.decoder(
    #             input_ids=generated,
    #             encoder_hidden_states=decoder_inputs,
    #             encoder_attention_mask=torch.ones_like(decoder_inputs[:, :, 0]),
    #         )

    #         # 预测下一个token
    #         next_token_logits = self.output_proj(
    #             decoder_outputs.last_hidden_state[:, -1:]
    #         )
    #         next_token = next_token_logits.argmax(dim=-1)

    #         # 拼接新token
    #         generated = torch.cat([generated, next_token], dim=1)

    #         # 检查是否生成结束标记
    #         if (next_token == self.tokenizer.eos_token_id).all():
    #             break
    #     generated_texts = self.tokenizer.batch_decode(
    #         generated,
    #         skip_special_tokens=True,  # 跳过特殊token如[PAD], [BOS], [EOS]等
    #         clean_up_tokenization_spaces=True,  # 清理tokenization产生的多余空格
    #     )
    #     return generated_texts

    def get_attention_weights(self):
        """获取注意力权重"""
        return self.attention_weights

    def analyze_attention(self, graph, node_ids):
        """分析特定节点的注意力分布"""
        h = self.encode_graph(graph)
        weights = self.get_attention_weights()

        # 分析第一层（4个头）的注意力
        layer1_weights = weights["layer1"]
        # 分析第二层（1个头）的注意力
        layer2_weights = weights["layer2"]

        return {"layer1": layer1_weights, "layer2": layer2_weights}

    def visualize_attention(self, attention_weights, node_labels):
        # 为每个头创建热力图
        for head in range(4):
            plt.figure(figsize=(10, 4))
            sns.heatmap(
                attention_weights["layer1"][head],
                xticklabels=node_labels,
                yticklabels=node_labels,
            )
            plt.title(f"Attention Head {head+1}")
            plt.show()
