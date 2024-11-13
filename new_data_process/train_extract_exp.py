##
import dgl
import json
import base64
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast
import yaml
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from build_base_graph_extract import (
    merge_graphs,
    ExplainableHeteroClassifier,
    batch_triplet_loss_with_neutral,
    save_texts_to_json,
)
from cal_scores import (
    f1_score,
    precision_score,
    recall_score,
    calculate_graph_bleu,
    is_best_model,
)
from path_ini import data_model_loader
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
import torch.multiprocessing as mp

# 多进程配置

import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

rel_names_short = [
    "Cause",
    "Condition",
    "Contrast",
    "Explanation",
    "Elaboration",
    "Attribution",
    "Background",
    "lexical",
]


def accuracy_score(y_true, y_pred):
    """
    计算准确率

    参数:
    y_true: 真实标签列表或数组
    y_pred: 预测标签列表或数组

    返回:
    float: 准确率（正确预测的样本比例）
    """
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 检查长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 必须具有相同的长度。")

    # 计算正确预测的数量
    correct_predictions = np.sum(y_true == y_pred)

    # 计算准确率
    accuracy = correct_predictions / len(y_true)

    return accuracy


def get_dataloader(dataset, batch_size, shuffle=True):
    """
    获取数据加载器

    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
    """
    dataset.load_batch_files(0)  # 一个文件
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # 设置为0，完全禁用多进程
        pin_memory=False,
        persistent_workers=False,
    )


def save_model(model, path, optimizer=None, scheduler=None, epoch=None, metrics=None):
    """
    保存模型和训练状态

    Args:
        model: 模型实例
        path: 保存路径
        optimizer: 优化器实例（可选）
        scheduler: 学习率调度器实例（可选）
        epoch: 当前轮次（可选）
        metrics: 评估指标（可选）
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": model.config if hasattr(model, "config") else None,
    }

    if optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        save_dict["epoch"] = epoch
    if metrics is not None:
        save_dict["metrics"] = metrics

    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 保存模型
    torch.save(save_dict, path)
    logging.info(f"Model saved to {path}")


def load_model(model, path, optimizer=None, scheduler=None):
    """
    加载模型和训练状态

    Args:
        model: 模型实例
        path: 加载路径
        optimizer: 优化器实例（可选）
        scheduler: 学习率调度器实例（可选）

    Returns:
        model: 加载后的模型
        optimizer: 加载后的优化器（如果提供）
        scheduler: 加载后的学习率调度器（如果提供）
        epoch: 保存时的轮次
        metrics: 保存时的评估指标
    """
    # 加载保存的字典
    checkpoint = torch.load(path)

    # 加载模型状态
    model.load_state_dict(checkpoint["model_state_dict"])

    # 如果保存了配置，并且模型有 config 属性，则加载配置
    if "config" in checkpoint and hasattr(model, "config"):
        model.config = checkpoint["config"]

    # 如果提供了优化器，加载其状态
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 如果提供了学习率调度器，加载其状态
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 获取保存的轮次和指标
    epoch = checkpoint.get("epoch")
    metrics = checkpoint.get("metrics")

    logging.info(f"Model loaded from {path}")
    return model, optimizer, scheduler, epoch, metrics


# 对node的text进行Base64解码
def decode_text_from_tensor(encoded_tensor):
    # 去除填充的空字节并转换为字节数据
    byte_data = bytes(encoded_tensor.tolist()).rstrip(b"\0")
    # 使用 Base64 解码还原原始文本
    decoded_text = base64.b64decode(byte_data).decode("utf-8")
    return decoded_text


def log_metrics(
    epoch, train_losses, eval_losses, eval_metrics_cli, eval_metrics_gen, stage
):
    """
    记录训练和评估指标

    Args:
        epoch: 当前轮次
        train_losses: 训练损失字典
        eval_losses: 评估指标字典
        stage: 训练阶段
    """
    # 构建日志消息
    log_msg = f"Epoch {epoch} ({stage})\n"

    # 添加训练损失
    log_msg += "Training Losses:\n"
    for loss_name, loss_value in train_losses.items():
        log_msg += f"  {loss_name}: {loss_value:.4f}\n"

    # 添加评估指标
    log_msg += "Evaluation Metrics:\n"
    for metric_name, metric_value in eval_losses.items():
        log_msg += f"  {metric_name}: {metric_value:.4f}\n"
    for metric_name, metric_value in eval_metrics_cli.items():
        log_msg += f"  {metric_name}: {metric_value:.4f}\n"
    for metric_name, metric_value in eval_metrics_gen.items():
        log_msg += f"  {metric_name}: {metric_value:.4f}\n"
    # 记录日志
    logging.info(log_msg)

    # 可选：使用 tensorboard 记录
    # if TENSORBOARD_AVAILABLE:
    #     for loss_name, loss_value in train_losses.items():
    #         writer.add_scalar(f'{stage}/train/{loss_name}', loss_value, epoch)
    #     for metric_name, metric_value in eval_losses.items():
    #         writer.add_scalar(f'{stage}/eval/{metric_name}', metric_value, epoch)


def get_current_metric(metrics, stage):
    """
    获取当前阶段的主要指标值

    Args:
        metrics: 评估指标字典
        stage: 训练阶段
    """
    if stage == "classification":
        return metrics.get("accuracy", 0)
    elif stage == "generation":
        return metrics.get("bleu", 0)
    else:  # joint
        return metrics.get("accuracy", 0) * 0.6 + metrics.get("bleu", 0) * 0.4


# 数据收集器类，用于跟踪训练过程中的指标
class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, metrics_dict):
        """更新指标"""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)

    def get_average(self, key):
        """获取特定指标的平均值"""
        values = self.metrics[key]
        return sum(values) / len(values) if values else 0

    def reset(self):
        """重置所有指标"""
        self.metrics.clear()

    def get_all_averages(self):
        """获取所有指标的平均值"""
        return {key: self.get_average(key) for key in self.metrics}


# 训练配置类
class TrainingConfig:
    def __init__(self, **kwargs):
        # 之前的配置项
        self.save_dir = kwargs.get("save_dir", "checkpoints")
        self.save_interval = kwargs.get(
            "save_interval", 5
        )  # 每多少个epoch保存一次检查点
        self.log_interval = kwargs.get("log_interval", 100)  # 每多少个batch记录一次日志
        self.use_tensorboard = kwargs.get("use_tensorboard", True)
        self.tensorboard_dir = kwargs.get("tensorboard_dir", "runs")

        # 验证相关
        self.eval_interval = kwargs.get("eval_interval", 1)  # 每多少个epoch进行一次验证

        # 模型相关
        self.model_config = kwargs.get("model_config", {})

        # 优化器相关
        self.optimizer_type = kwargs.get("optimizer_type", "adamw")
        self.scheduler_type = kwargs.get("scheduler_type", "linear_warmup")

    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置实例"""
        return cls(**config_dict)

    def to_dict(self):
        """将配置转换为字典"""
        return self.__dict__


def collate_fn(batch):
    (
        g_premises,
        g_hypotheses,
        lexical_chains,
        hyp_emb_threes,
        exp_text_threes,
        hyp_text_threes,
        label_tensors,
    ) = zip(*batch)

    # 处理假设嵌入
    hyp_emb_threes = torch.stack(
        [
            torch.stack(
                [
                    torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
                    for emb in sample
                ]
            )
            for sample in hyp_emb_threes
        ]
    )

    # 处理文本数据
    exp_text_threes = list(exp_text_threes)
    hyp_text_threes = list(hyp_text_threes)
    list_0, list_1, list_2 = zip(*exp_text_threes)
    hyp_list_0, hyp_list_1, hyp_list_2 = zip(*hyp_text_threes)

    # # 对图进行批处理
    # batched_g_premises = dgl.batch(list(g_premises))
    # batched_g_hypotheses = dgl.batch(list(g_hypotheses))

    # 处理标签张量 - 不再stack，而是保持列表形式
    batched_labels = {
        "premise1": {
            "entailment": [
                labels["premise1"]["entailment"] for labels in label_tensors
            ],
            "neutral": [labels["premise1"]["neutral"] for labels in label_tensors],
            "contradiction": [
                labels["premise1"]["contradiction"] for labels in label_tensors
            ],
        },
        "premise2": {
            "entailment": [
                labels["premise2"]["entailment"] for labels in label_tensors
            ],
            "neutral": [labels["premise2"]["neutral"] for labels in label_tensors],
            "contradiction": [
                labels["premise2"]["contradiction"] for labels in label_tensors
            ],
        },
    }

    return (
        list(g_premises),
        list(g_hypotheses),
        list(lexical_chains),
        hyp_emb_threes,
        (list(list_0), list(list_1), list(list_2)),
        (list(hyp_list_0), list(hyp_list_1), list(hyp_list_2)),
        batched_labels,
    )


def process_batch(model, batch_data, device, task, stage="train", optimizer=None):
    """处理一个batch的通用逻辑"""
    graph1, graph2, lexical_chain, hyp_emb, exp_text, hyp_text, edu_labels = batch_data
    batch_loss = 0
    # 合并图
    combined_graphs = [
        merge_graphs(g_p1, g_p2, lc, rel_names_short)
        for g_p1, g_p2, lc in zip(graph1, graph2, lexical_chain)
    ]
    combined_graphs = dgl.batch(combined_graphs).to(device)
    for g1 in graph1:
        if "node_id" in g1.ndata:
            g1.ndata.pop("node_id")
    for g2 in graph2:
        if "node_id" in g2.ndata:
            g2.ndata.pop("node_id")
    graph1 = dgl.batch(graph1).to(device)
    graph2 = dgl.batch(graph2).to(device)

    hyp_tensor_embeddings = hyp_emb.to(device)
    positive = hyp_tensor_embeddings[:, 0, :]
    neutral = hyp_tensor_embeddings[:, 1, :]
    negative = hyp_tensor_embeddings[:, 2, :]

    # 获取图信息
    node_feats, attention_weights, graph_repr = model.graph_info(combined_graphs)

    batch_metrics = {
        "losses": defaultdict(float),
        "predictions": [],
        "labels": [],
        "generation_metrics": defaultdict(float),
        "generation_metrics_acc": defaultdict(float),
    }

    # 分类任务
    if task == "classification" or task == "joint":
        # logging.info("Processing classification task")
        predicted_cli_batch = []
        labels_cli_batch = []
        triplet_loss = batch_triplet_loss_with_neutral(
            graph_repr, hyp_tensor_embeddings
        )
        classification_losses = []

        for embeddings, targets, _, label in [
            (
                positive,
                torch.zeros(len(hyp_emb), dtype=torch.long, device=device),
                None,
                "entailment",
            ),
            (
                neutral,
                torch.ones(len(hyp_emb), dtype=torch.long, device=device),
                None,
                "neutral",
            ),
            (
                negative,
                torch.full((len(hyp_emb),), 2, dtype=torch.long, device=device),
                None,
                "contradiction",
            ),
        ]:
            cli_logits = model.classify(graph_repr, embeddings)
            cls_loss = F.cross_entropy(cli_logits, targets)
            predicted = torch.argmax(cli_logits, dim=-1).tolist()
            classification_losses.append(cls_loss)
            predicted_cli_batch.extend(predicted)
            labels_cli_batch.extend(targets.cpu().numpy())

        classification_loss = sum(classification_losses)
        batch_loss = classification_loss + triplet_loss

        batch_metrics["losses"]["triplet_loss"] += triplet_loss.item()
        batch_metrics["losses"]["cls_loss"] += classification_loss.item()
        batch_metrics["predictions"].extend(predicted_cli_batch)
        batch_metrics["labels"].extend(labels_cli_batch)

    # 生成任务
    if task == "generation" or task == "joint":
        # logging.info("Processing generation task")
        predicted_nodes_features = {
            relation: {"graph1": [], "graph2": []}
            for relation in ["entailment", "neutral", "contradiction"]
        }

        for relation, hyp_emb in zip(
            ["entailment", "neutral", "contradiction"],
            [positive, neutral, negative],
        ):
            edu_labels["premise1"][relation] = [
                tensor.to(device) for tensor in edu_labels["premise1"][relation]
            ]
            edu_labels["premise2"][relation] = [
                tensor.to(device) for tensor in edu_labels["premise2"][relation]
            ]
            premise_labels = edu_labels["premise1"][relation]
            premise2_labels = edu_labels["premise2"][relation]

            p1_node_logits, p1_attn = model.extract_explanation(graph1, hyp_emb)
            p2_node_logits, p2_attn = model.extract_explanation(graph2, hyp_emb)

            total_len_p1 = sum(t.size(0) for t in premise_labels)
            total_positives_p1 = sum((t == 1).sum().item() for t in premise_labels)

            total_len_p2 = sum(t.size(0) for t in premise2_labels)
            total_positives_p2 = sum((t == 1).sum().item() for t in premise2_labels)

            class_weights_p1 = torch.tensor(
                [1.0, total_len_p1 / (2 * max(1, total_positives_p1))]
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights_p1).to(
                device
            )  # 为了平衡正负样本

            extract_loss1 = criterion(
                p1_node_logits, torch.cat(premise_labels).to(device)
            )

            class_weights_p2 = torch.tensor(
                [1.0, total_len_p2 / (2 * max(1, total_positives_p2))]
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights_p2).to(device)

            extract_loss2 = criterion(
                p2_node_logits, torch.cat(premise2_labels).to(device)
            )

            pair_loss = extract_loss1 + extract_loss2

            batch_loss += pair_loss
            batch_metrics["losses"]["gen_loss"] += (
                extract_loss1.item() + extract_loss2.item()
            )

            if stage != "train":
                # 计算生成任务的指标

                p1_predictions = torch.argmax(p1_node_logits, dim=-1)
                p2_predictions = torch.argmax(p2_node_logits, dim=-1)

                # 初始化存储结构
                g1_target_texts = []
                g2_target_texts = []
                g1_pred_texts = []
                g2_pred_texts = []

                # 处理graph1
                g1_cumsum = 0
                for batch_idx, batch_label in enumerate(premise_labels):
                    # 获取目标文本
                    positive_indices = batch_label.nonzero().squeeze(-1)
                    if positive_indices.dim() == 0 and positive_indices.nelement() == 1:
                        positive_indices = positive_indices.unsqueeze(0)
                    adjusted_indices = positive_indices + g1_cumsum
                    edus_1 = [
                        graph1.ndata["text_encoded"][i.item()] for i in adjusted_indices
                    ]
                    edus_1 = [decode_text_from_tensor(text) for text in edus_1]
                    g1_target_texts.append(edus_1)
                    # 获取预测文本
                    batch_predictions = p1_predictions[
                        g1_cumsum : g1_cumsum + batch_label.size(0)
                    ]
                    pred_indices = batch_predictions.eq(1).nonzero().squeeze(-1)
                    if pred_indices.dim() == 0 and pred_indices.nelement() == 1:
                        pred_indices = pred_indices.unsqueeze(0)
                    adjusted_pred_indices = pred_indices + g1_cumsum
                    edus1_pre = [
                        graph1.ndata["text_encoded"][i.item()]
                        for i in adjusted_pred_indices
                    ]
                    edus1_pre = [decode_text_from_tensor(text) for text in edus1_pre]
                    g1_pred_texts.append(edus1_pre)
                    g1_cumsum += batch_label.size(0)

                # 处理graph2
                g2_cumsum = 0
                for batch_idx, batch_label in enumerate(premise2_labels):
                    # 获取目标文本
                    positive_indices = batch_label.nonzero().squeeze(-1)
                    if positive_indices.dim() == 0 and positive_indices.nelement() == 1:
                        positive_indices = positive_indices.unsqueeze(0)
                    adjusted_indices = positive_indices + g2_cumsum
                    edus_2 = [
                        graph2.ndata["text_encoded"][i.item()] for i in adjusted_indices
                    ]
                    edus_2 = [decode_text_from_tensor(text) for text in edus_2]
                    g2_target_texts.append(edus_2)

                    # 获取预测文本
                    batch_predictions = p2_predictions[
                        g2_cumsum : g2_cumsum + batch_label.size(0)
                    ]
                    pred_indices = batch_predictions.eq(1).nonzero().squeeze(-1)
                    if pred_indices.dim() == 0 and pred_indices.nelement() == 1:
                        pred_indices = pred_indices.unsqueeze(0)
                    adjusted_pred_indices = pred_indices + g2_cumsum
                    edus_2_pre = [
                        graph2.ndata["text_encoded"][i.item()]
                        for i in adjusted_pred_indices
                    ]
                    edus_2_pre = [decode_text_from_tensor(text) for text in edus_2_pre]
                    g2_pred_texts.append(edus_2_pre)
                    g2_cumsum += batch_label.size(0)

                # 计算每个图的BLEU值
                g1_bleu_list = [
                    calculate_graph_bleu(g1_target_text, g1_pred_text)
                    for g1_target_text, g1_pred_text in zip(
                        g1_target_texts, g1_pred_texts
                    )
                ]
                g2_bleu_list = [
                    calculate_graph_bleu(g2_target_text, g2_pred_text)
                    for g2_target_text, g2_pred_text in zip(
                        g2_target_texts, g2_pred_texts
                    )
                ]
                g1_bleu = sum(g1_bleu_list) / len(g1_bleu_list) if g1_bleu_list else 0
                g2_bleu = sum(g2_bleu_list) / len(g2_bleu_list) if g2_bleu_list else 0

                # 存储结果
                predicted_nodes_features[relation]["graph1"] = {
                    # "target_texts": g1_target_texts,
                    # "pred_texts": g1_pred_texts,
                    "bleu": g1_bleu
                }

                predicted_nodes_features[relation]["graph2"] = {
                    # "target_texts": g2_target_texts,
                    # "pred_texts": g2_pred_texts,
                    "bleu": g2_bleu
                }

                # 获取当前图中预测为1的节点索引
                batch_metrics["generation_metrics_acc"][f"{relation}_p1_acc"] = (
                    (p1_predictions == torch.cat(premise_labels)).float().mean().item()
                )
                batch_metrics["generation_metrics"][f"{relation}_p1_bleu"] = g1_bleu

                batch_metrics["generation_metrics_acc"][f"{relation}_p2_acc"] = (
                    (p2_predictions == torch.cat(premise2_labels)).float().mean().item()
                )
                batch_metrics["generation_metrics"][f"{relation}_p2_bleu"] = g2_bleu

    if stage == "train" and optimizer is not None:
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return batch_metrics


def train_epoch(model, dataloader, optimizer, task, device):
    """训练一个epoch"""
    model.train()
    epoch_losses = defaultdict(float)

    for batch_data in dataloader:
        batch_metrics = process_batch(
            model, batch_data, device, task, "train", optimizer
        )
        for loss_name, loss_value in batch_metrics["losses"].items():
            epoch_losses[loss_name] += loss_value

    # 计算平均损失
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)

    return epoch_losses


def eval_epoch(model, dataloader, task, device):
    """评估一个epoch"""
    model.eval()
    epoch_losses = defaultdict(float)
    all_predictions = []
    all_labels = []
    classification_metrics = {}
    generation_avg_metrics = {}
    generation_metrics = defaultdict(float)

    with torch.no_grad():
        for batch_data in dataloader:
            batch_metrics = process_batch(model, batch_data, device, task, "eval")

            # 累积损失
            for loss_name, loss_value in batch_metrics["losses"].items():
                epoch_losses[loss_name] += loss_value

            if task == "classification" or task == "joint":
                # 累积预测结果
                all_predictions.extend(batch_metrics["predictions"])
                all_labels.extend(batch_metrics["labels"])
            if task == "generation" or task == "joint":
                # 累积生成指标
                for key, value in batch_metrics["generation_metrics"].items():
                    generation_metrics[key] += value
                for key, value in batch_metrics["generation_metrics_acc"].items():
                    generation_metrics[key] += value
    # 计算平均损失
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    if task == "classification" or task == "joint":
        # 计算分类指标
        classification_metrics = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1": f1_score(all_labels, all_predictions, average="macro"),
            "precision": precision_score(all_labels, all_predictions, average="macro"),
            "recall": recall_score(all_labels, all_predictions, average="macro"),
        }

    # 计算生成任务的平均指标
    if task == "generation" or task == "joint":
        for key in generation_metrics:
            generation_avg_metrics[key] = generation_metrics[key] / len(dataloader)
        logging.info(f"in eval generation_avg_metrics: {generation_avg_metrics}")
        generation_sum_metrics = {
            "pair_acc": sum(
                v for k, v in generation_avg_metrics.items() if "acc" in k.lower()
            )
            / sum(1 for k in generation_avg_metrics if "acc" in k.lower()),
            "pair_bleu": sum(
                v for k, v in generation_avg_metrics.items() if "bleu" in k.lower()
            )
            / sum(1 for k in generation_avg_metrics if "bleu" in k.lower()),
        }

    return {
        "losses": epoch_losses,
        "classification_metrics": classification_metrics,
        "generation_sum_metrics": generation_sum_metrics,
    }


def represent_torch_device(dumper, data):

    return dumper.represent_scalar("!torch.device", str(data))


def main():
    """
    主训练函数
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )

    try:

        # 初始化模型和数据集
        logging.info(f"Using device: {device}")

        config, train_dataset, dev_dataset, test_dataset = data_model_loader(device)

        train_loader = get_dataloader(train_dataset, config.batch_size)
        dev_loader = get_dataloader(dev_dataset, config.batch_size)
        test_loader = get_dataloader(test_dataset, config.batch_size)
        # 保存配置
        os.makedirs(config.save_dir, exist_ok=True)
        yaml.add_representer(torch.device, represent_torch_device)

        with open(os.path.join(config.save_dir, "config.yaml"), "w") as f:
            yaml.dump(config.to_dict(), f)
        model = ExplainableHeteroClassifier(
            in_dim=config.model_config["in_dim"],
            hidden_dim=config.model_config["hidden_dim"],
            n_classes=config.model_config["n_classes"],
            rel_names=config.model_config["rel_names"],
            device=config.model_config["device"],
        ).to(device)
        model = model.to(device)
        # 设置优化器和调度器
        optimizer = torch.optim.Adam(model.parameters(), lr=config.classifier_lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )
        start_epoch = 0
        model, optimizer, scheduler, start_epoch, metrics = load_model(
            model,
            "/mnt/nlp/yuanmengying/ymy/new_data_process/checkpoints/experiment1/best_joint_model.pt",
            optimizer,
            scheduler,
        )

        # 初始化指标跟踪器
        metrics_tracker = MetricsTracker()

        # 训练循环
        config.mode = "train"
        config.stage = "generation"
        config.epochs = 60
        stage = config.stage
        logging.info(f"Starting {config.mode}  ==  {stage} task...")
        best_classification_f1 = 0
        best_generation_f1 = 0
        best_joint_f1 = 0
        for epoch in range(start_epoch, config.epochs):
            # 训练一个epoch
            train_losses = train_epoch(
                model, train_loader, optimizer, task=config.stage, device=device
            )
            logging.info(f"Epoch {epoch} train losses: {train_losses}")
            # 评估
            eval_results = eval_epoch(
                model, dev_loader, task=config.stage, device=device
            )
            logging.info(f"Epoch {epoch} eval losses: {eval_results['losses']}")
            # 更新学习率
            if scheduler is not None:
                sum_eval_loss = sum(eval_results["losses"].values())
                scheduler.step(sum_eval_loss)

            # 记录指标
            metrics_tracker.update(
                {
                    **{f"train_{k}": v for k, v in train_losses.items()},
                    **{f"eval_{k}": v for k, v in eval_results["losses"].items()},
                    **{
                        f"eval_{k}": v
                        for k, v in eval_results["classification_metrics"].items()
                    },
                    **{
                        f"eval_{k}": v
                        for k, v in eval_results["generation_sum_metrics"].items()
                    },
                }
            )

            # 记录日志
            log_metrics(
                epoch,
                train_losses,
                eval_results["losses"],
                eval_results["classification_metrics"],
                eval_results["generation_sum_metrics"],
                stage,
            )
            if stage == "classification" and is_best_model(
                eval_results, best_classification_f1, stage
            ):  # 分类任务，用f1指标
                best_classification_f1 = eval_results["classification_metrics"]["f1"]
                model_path = os.path.join(
                    config.save_dir, f"best_classification_model.pt"
                )
                save_model(
                    model,
                    model_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=eval_results["classification_metrics"],
                )
                logging.info(
                    f"New best classification model saved with metric: {best_classification_f1:.4f}"
                )

            elif stage == "generation" and is_best_model(
                eval_results, best_generation_f1, stage
            ):
                best_generation_f1 = eval_results["generation_sum_metrics"]["pair_bleu"]
                model_path = os.path.join(config.save_dir, f"best_generation_model.pt")
                save_model(
                    model,
                    model_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=eval_results,
                )
                logging.info(
                    f"New best generation model saved with metric: {best_generation_f1:.4f}"
                )
            elif stage == "joint" and is_best_model(eval_results, best_joint_f1, stage):
                best_joint_f1 = (
                    eval_results["generation_sum_metrics"]["pair_bleu"] * 0.4
                    + eval_results["classification_metrics"]["f1"] * 0.6
                )
                model_path = os.path.join(config.save_dir, f"best_joint_model.pt")
                save_model(
                    model,
                    model_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=eval_results["generation_sum_metrics"],
                )
        # 阶段结束，记录最终指标
        stage_metrics = metrics_tracker.get_all_averages()
        logging.info(f"\n{stage} training completed.")
        logging.info("Final metrics:")
        for metric_name, value in stage_metrics.items():
            logging.info(f"  {metric_name}: {value:.4f}")

        # 重置指标跟踪器，准备下一阶段
        metrics_tracker.reset()

        # 训练完成，记录最终结果
        logging.info("\nTraining completed.")

        #     # 测试
        logging.info("\nTesting...")
        test_metrics = eval_epoch(model, test_loader, task=config.stage, device=device)
        logging.info("\nTest metrics:")
        for metric_name, value in test_metrics["classification_metrics"].items():
            logging.info(f"  {metric_name}: {value:.4f}")
        for metric_name, value in test_metrics["generation_sum_metrics"].items():
            logging.info(f"  {metric_name}: {value:.4f}")

        # 记录日志
        log_metrics(
            epoch,
            train_losses,
            test_metrics["losses"],
            test_metrics["classification_metrics"],
            test_metrics["generation_sum_metrics"],
            stage,
        )
    # except Exception as e:
    #     logging.error(f"Training failed with error: {str(e)}")
    #     logging.error(traceback.format_exc())
    #     raise
    finally:
        # 清理资源
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
