import dgl
import torch
from datetime import datetime
from sklearn.metrics import f1_score
from build_base_graph import build_graph
from dgl.dataloading import GraphDataLoader
from data_loader import RSTDataset
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
import os
import numpy as np


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"DGL version: {dgl.__version__}")
import logging
from build_base_graph_new import merge_graphs

# from data_loader import create_dataloader
# from build_base_graph import HeteroClassifier
from build_base_graph_3 import HeteroClassifier

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import torch.nn.functional as F


def batch_triplet_loss_with_neutral(anchor, hyp, margin=1.0, neutral_weight=0.5):
    """
    计算批量三元组损失，支持 positive 和 negative 的维度为 [batch_size, 3, embedding_dim]。

    参数：
    - anchor: 前提的嵌入 (batch_size, embedding_dim)，形状为 [16, 1024]。
    - hyp: 假设的嵌入 (batch_size, 3, embedding_dim)，形状为 [16, 3, 1024]，包括 positive, neutral 和 negative。
    - margin: 三元组损失的间隔。
    - neutral_weight: 中立假设损失的权重。

    返回：
    - total_loss: 批量的总损失。
    """
    # 分别提取 positive（蕴含），neutral（中立），negative（矛盾）的嵌入
    positive = hyp[:, 0, :]  # [16, 1024] 蕴含
    neutral = hyp[:, 1, :]  # [16, 1024] 中立
    negative = hyp[:, 2, :]  # [16, 1024] 矛盾

    # 计算 anchor 和 positive（蕴含）的距离
    dist_pos = F.pairwise_distance(anchor, positive, p=2)  # [batch_size]

    # 计算 anchor 和 negative（矛盾）的距离
    dist_neg = F.pairwise_distance(anchor, negative, p=2)  # [batch_size]

    # 计算 anchor 和 neutral（中立）的距离
    dist_neutral = F.pairwise_distance(anchor, neutral, p=2)  # [batch_size]

    # 三元组损失：正样本与 Anchor 应更接近
    loss_triplet = torch.clamp(dist_pos - dist_neg + margin, min=0).mean()

    # 中立损失：Neutral 应介于 Positive 和 Negative 之间
    loss_neutral = (
        torch.clamp(dist_pos - dist_neutral, min=0)
        + torch.clamp(dist_neutral - dist_neg, min=0)
    ).mean()

    # 总损失：三元组损失 + 中立损失
    total_loss = loss_triplet + neutral_weight * loss_neutral

    return total_loss


def evaluate(dataset, model, device, batch_size=256):
    model.eval()  # 切换到评估模式
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        # 遍历文件批次
        for batch_num in range(dataset.total_batches):
            if batch_num % 10 == 0:
                print(f"Evaluating batch {batch_num + 1}/{dataset.total_batches}...")

            # 加载当前文件批次并构建图
            dataset.load_batch_files(batch_num)

            # 创建 DataLoader 对当前文件批次的数据进行迭代
            dataloader = GraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # 评估时不需要打乱顺序
                collate_fn=collate_fn,
            )

            for index, (g_premise, g_hypothesis, lexical_chain, hyp_emb) in enumerate(
                dataloader
            ):
                # 将数据移动到 GPU
                true_labels = [0, 1, 2] * len(hyp_emb)
                labels = torch.tensor(true_labels).to(device)

                # 合并图结构
                combined_graphs = [
                    merge_graphs(g_p, g_h, lc, rel_names_short)
                    for g_p, g_h, lc in zip(g_premise, g_hypothesis, lexical_chain)
                ]
                # g_premise = dgl.batch(g_premise).to(device)
                # g_hypothesis = dgl.batch(g_hypothesis).to(device)
                combined_graphs = dgl.batch(combined_graphs).to(device)
                # 前向传播
                hg_combined = model(combined_graphs)
                hyp_tensor_embeddings = hyp_emb.to(device)
                # 计算三元组损失
                triplet_loss = batch_triplet_loss_with_neutral(
                    hg_combined, hyp_tensor_embeddings
                )
                positive = hyp_tensor_embeddings[:, 0, :]  # [16, 1024] 蕴含
                neutral = hyp_tensor_embeddings[:, 1, :]  # [16, 1024] 中立
                negative = hyp_tensor_embeddings[:, 2, :]  # [16, 1024] 矛盾

                # # 对蕴含假设进行分类
                logits_entailment = model.classify_hypothesis(hg_combined, positive)
                predicted_0 = torch.argmax(logits_entailment, dim=-1)
                targets = torch.zeros(len(hyp_emb), dtype=torch.long, device=device)
                loss_entailment = F.cross_entropy(logits_entailment, targets)

                # # 对中立假设进行分类
                logits_nerutral = model.classify_hypothesis(hg_combined, neutral)
                predicted_1 = torch.argmax(logits_nerutral, dim=-1)
                targets = torch.ones(len(hyp_emb), dtype=torch.long, device=device)
                loss_neutral = F.cross_entropy(logits_nerutral, targets)
                # 对矛盾假设进行分类
                logits_conflict = model.classify_hypothesis(hg_combined, negative)
                predicted_2 = torch.argmax(
                    logits_conflict, dim=-1
                )  # 不用F.softmax(logits, dim=-1)
                targets = torch.full(
                    (len(hyp_emb),), 2, dtype=torch.long, device=device
                )
                loss_contradiction = F.cross_entropy(logits_conflict, targets)
                # 总分类损失：蕴含 + 中立 + 矛盾
                classification_loss = (
                    loss_entailment + loss_neutral + loss_contradiction
                )

                # 最终的总损失：三元组损失 + 分类损失
                loss = triplet_loss + classification_loss
                # 记录损失
                total_loss += loss.item()

                stacked_tensor = torch.stack([predicted_0, predicted_1, predicted_2])
                transposed_tensor = stacked_tensor.T
                predicted_result = transposed_tensor.flatten()
                total_correct += (predicted_result == labels).sum().item()
                total_samples += labels.size(0)
                # 收集所有预测和真实标签
                all_predictions.extend(predicted_result.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / dataset.total_batches
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    # logging.info(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}, F1 Score: {f1}")
    # return accuracy, f1
    return avg_loss, accuracy, f1


def collate_fn(batch):
    g_premises, g_hypotheses, lexical_chains, hyp_emb_threes = zip(*batch)
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
    return (
        list(g_premises),
        list(g_hypotheses),
        list(lexical_chains),
        hyp_emb_threes,
    )


def train(
    dataset,
    model,
    dev_dataset,
    epochs=10,
    lr=0.001,  # 调整学习率
    save_path="best_model.pth",
    batch_size=256,
    accumulation_steps=4,
    device=device,
    use_distributed=False,
):
    if use_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # best_dev_loss = float("inf")
    best_dev_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_num in range(0, dataset.total_batches):
            if batch_num % 10 == 0:
                print(f"Loading batch {batch_num + 1}/{dataset.total_batches}...")

            dataset.load_batch_files(batch_num)
            dataloader = GraphDataLoader(
                dataset,
                batch_size=batch_size,  # 256
                shuffle=True,
                collate_fn=collate_fn,
            )
            for (
                g_premise,
                g_premise2,
                lexical_chain,
                hyp_emb,
            ) in dataloader:  # 这里的是一个batch_size的
                with autocast():
                    # 合并图结构
                    combined_graphs = [
                        merge_graphs(g_p1, g_p2, lc, rel_names_short)
                        for g_p1, g_p2, lc in zip(g_premise, g_premise2, lexical_chain)
                    ]

                    combined_graphs = dgl.batch(combined_graphs).to(device)
                    # 获取大前提的合并图嵌入
                    hg_combined = model(combined_graphs)

                    hyp_tensor_embeddings = hyp_emb.to(device)

                    # 计算三元组损失
                    triplet_loss = batch_triplet_loss_with_neutral(
                        hg_combined, hyp_tensor_embeddings
                    )
                    positive = hyp_tensor_embeddings[:, 0, :]  # [16, 1024] 蕴含
                    neutral = hyp_tensor_embeddings[:, 1, :]  # [16, 1024] 中立
                    negative = hyp_tensor_embeddings[:, 2, :]  # [16, 1024] 矛盾
                    # # 计算三个假设的分类损失
                    # # 对蕴含假设进行分类
                    logits_entailment = model.classify_hypothesis(hg_combined, positive)
                    targets = torch.zeros(len(hyp_emb), dtype=torch.long, device=device)
                    loss_entailment = F.cross_entropy(logits_entailment, targets)

                    # # 对中立假设进行分类
                    logits_neutral = model.classify_hypothesis(hg_combined, neutral)
                    targets = torch.ones(len(hyp_emb), dtype=torch.long, device=device)
                    loss_neutral = F.cross_entropy(logits_neutral, targets)

                    # 对矛盾假设进行分类
                    logits_contradiction = model.classify_hypothesis(
                        hg_combined, negative
                    )
                    targets = torch.full(
                        (len(hyp_emb),), 2, dtype=torch.long, device=device
                    )

                    loss_contradiction = F.cross_entropy(logits_contradiction, targets)
                    # 总分类损失：蕴含 + 中立 + 矛盾
                    classification_loss = (
                        loss_entailment + loss_neutral + loss_contradiction
                    )

                    # 最终的总损失：三元组损失 + 分类损失
                    loss = triplet_loss + classification_loss
                    # 记录损失
                    total_loss += loss.item()

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        dev_loss, dev_accuracy, dev_f1 = evaluate(dev_dataset, model, device)
        logging.info(
            f"Epoch {epoch+1}/{epochs}， "
            f"Validation Loss: {dev_loss}, Validation Accuracy: {dev_accuracy}, dev_f1 Score: {dev_f1}"
        )
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_save_path = f"model6{save_path}_epoch{epoch+1}_{current_time}.pth"
            if use_distributed:
                if dist.get_rank() == 0:
                    torch.save(model.module.state_dict(), full_save_path)
                    logging.info(f"Model saved at epoch {epoch+1}: {full_save_path}")
            else:
                torch.save(model.state_dict(), full_save_path)
                logging.info(f"Model saved at epoch {epoch+1}: {full_save_path}")
        scheduler.step(dev_loss)
        # 在需要查看学习率的地方使用
        current_lr = scheduler.get_last_lr()
        print(f"Current learning rate: {current_lr}")
    avg_loss = total_loss / dataset.total_batches
    logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")


def test(dataset, model, device, batch_size=128):
    model.eval()  # 切换到评估模式
    model.to(device)
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        # 遍历文件批次
        for batch_num in range(dataset.total_batches):
            if batch_num % 10 == 0:
                print(f"Testing batch {batch_num + 1}/{dataset.total_batches}...")

            # 加载当前文件批次并构建图
            dataset.load_batch_files(batch_num)

            # 创建 DataLoader 对当前文件批次的数据进行迭代
            dataloader = GraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # 测试时不需要打乱顺序
                collate_fn=collate_fn,
            )

            for g_premise, g_hypothesis, lexical_chain, hyp_emb in dataloader:
                true_labels = [0, 1, 2] * len(hyp_emb)
                labels = torch.tensor(true_labels).to(device)

                # 合并图结构
                combined_graphs = [
                    merge_graphs(g_p, g_h, lc, rel_names_short)
                    for g_p, g_h, lc in zip(g_premise, g_hypothesis, lexical_chain)
                ]

                combined_graphs = dgl.batch(combined_graphs).to(device)
                # 前向传播
                hg_combined = model(combined_graphs)
                hyp_tensor_embeddings = hyp_emb.to(device)

                positive = hyp_tensor_embeddings[:, 0, :]  # [16, 1024] 蕴含
                neutral = hyp_tensor_embeddings[:, 1, :]  # [16, 1024] 中立
                negative = hyp_tensor_embeddings[:, 2, :]  # [16, 1024] 矛盾

                logits_entailment = model.classify_hypothesis(hg_combined, positive)
                predicted_0 = torch.argmax(logits_entailment, dim=-1)
                logits_nerutral = model.classify_hypothesis(hg_combined, neutral)
                predicted_1 = torch.argmax(logits_nerutral, dim=-1)
                logits_conflict = model.classify_hypothesis(hg_combined, negative)
                predicted_2 = torch.argmax(
                    logits_conflict, dim=-1
                )  # 不用F.softmax(logits, dim=-1)

                stacked_tensor = torch.stack([predicted_0, predicted_1, predicted_2])
                transposed_tensor = stacked_tensor.T
                predicted_result = transposed_tensor.flatten()
                total_correct += (predicted_result == labels).sum().item()
                total_samples += labels.size(0)

    # 计算准确率和F1
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    logging.info(f"Test Accuracy: {accuracy}, F1 Score: {f1}")
    return accuracy, f1


if __name__ == "__main__":
    # 加载数据和模型

    batch_file_size = 1

    train_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/train/train1/new_rst_result.jsonl"
    model_output_path = (
        r"/mnt/nlp/yuanmengying/nli_data_generate/all_generated_hypothesis.json"
    )
    model_output_emb_path = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/train/hyp/hypothesis_embeddings.npz"
    train_lexical_chains_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_infos/train"
    )
    train_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/train/pre"
    train_pair_graph = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_pairs/train"
    )
    logging.info("Processing train data")
    train_dataset = RSTDataset(
        train_rst_path,
        model_output_path,
        model_output_emb_path,
        train_lexical_chains_path,
        train_embedding_file,
        batch_file_size,
        save_dir=train_pair_graph,
    )

    logging.info("Processing dev data")

    dev_rst_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/dev/dev1/new_rst_result.jsonl"
    )
    dev_model_output_path = (
        r"/mnt/nlp/yuanmengying/nli_data_generate/valid_all_generated_hypothesis.json"
    )
    dev_model_output_emb_path = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/dev/hyp/hypothesis_embeddings.npz"
    dev_lexical_chains_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_infos/dev"
    )
    dev_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/dev/pre"
    dev_pair_graph = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_pairs/dev"
    dev_dataset = RSTDataset(
        dev_rst_path,
        dev_model_output_path,
        dev_model_output_emb_path,
        dev_lexical_chains_path,
        dev_embedding_file,
        batch_file_size,
        save_dir=dev_pair_graph,
    )

    logging.info("Processing test data")

    num_rels = 20
    rel_names = [
        "Cause",
        "Condition",
        "Contrast",
        "Explanation",
        "Elaboration",
        "Attribution",
        "Background",
        "Joint",
        "Topic-Comment",
        "Comparison",
        "Condition",
        "Contrast",
        "Evaluation",
        "Topic-Change",
        "Summary",
        "Manner-Means",
        "Temporal",
        "TextualOrganization",
        "Enablement",
        "Same-Unit",
        "span",  # 可以考虑去掉，因为关系不密切
        "lexical",  # 词汇链
    ]
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

    model2 = HeteroClassifier(
        in_dim=1024,
        hidden_dim=1024,
        n_classes=3,
        rel_names=rel_names,
        rel_names_short=rel_names_short,
    )
    full_save_path = "/mnt/nlp/yuanmengying/ymy/new_data_process/model6best_model.pth_epoch56_20241026_004543.pth"  # 5+20+30的
    model2.load_state_dict(torch.load(full_save_path))
    # 训练模型
    train(
        train_dataset,
        model2,
        dev_dataset,
        epochs=60,
        device=device,
        # use_distributed=True,
    )
