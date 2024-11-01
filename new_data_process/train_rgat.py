import dgl
import json
import torch
import numpy as np
from rouge import Rouge
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import f1_score
from dgl.dataloading import GraphDataLoader
from data_loader_exp import RSTDataset
from torch.cuda.amp import autocast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BartTokenizer
from transformers import T5Tokenizer
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns

decode_name = r"/mnt/nlp/yuanmengying/models/t5-large"
# tokenizer = BartTokenizer.from_pretrained("/mnt/nlp/yuanmengying/models/bart-large")
tokenizer = T5Tokenizer.from_pretrained(decode_name)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"DGL version: {dgl.__version__}")
import logging
from build_base_graph_rgat import merge_graphs
import traceback
import gc

# from data_loader import create_dataloader
# from build_base_graph import HeteroClassifier
from build_base_graph_rgat import ExplainableHeteroClassifier

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def save_texts_to_json(all_generated_text, all_golden_text, filename):
    """
    将两个列表保存到一个JSON文件中，每个元素是一个包含"generated"和"target"的字典。

    参数:
    - all_generated_text: 包含生成文本的列表。
    - all_golden_text: 包含目标文本的列表。
    - filename: 要保存的JSON文件名。
    """
    # 确保两个列表长度相同
    if len(all_generated_text) != len(all_golden_text):
        raise ValueError("The lists must be of the same length.")

    # 合并成一个列表，其中每个元素是一个字典
    combined_list = [
        {"generated": gen, "target": gold}
        for gen, gold in zip(all_generated_text, all_golden_text)
    ]

    # 将结果写入JSON文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=4)

    print(f"Data has been written to {filename}")


rouge = Rouge()

smoothing_function = SmoothingFunction().method1


def cal_bleu_rouge(generated_explanations, exp_text):

    bleu_scores = []
    rouge_scores = []
    for gen_exp, ref_exp in zip(generated_explanations, exp_text):
        # 使用BART tokenizer进行分词
        gen_tokens = tokenizer.tokenize(gen_exp)
        ref_tokens = tokenizer.tokenize(ref_exp)

        # 计算BLEU分数
        bleu_score = sentence_bleu(
            [ref_tokens], gen_tokens, smoothing_function=smoothing_function
        )
        bleu_scores.append(bleu_score)

        # 计算ROUGE分数
        try:
            rouge_score = rouge.get_scores(gen_exp, ref_exp)
            rouge_scores.append(rouge_score[0])
        except Exception as e:
            print(f"计算ROUGE分数时出错: {e}")
            rouge_scores.append(
                {
                    "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0},
                }
            )

    return bleu_scores, rouge_scores


def calculate_average_scores(bleu_scores, rouge_scores):
    # 计算BLEU平均值
    avg_bleu = np.mean(bleu_scores)

    # 计算ROUGE平均值
    avg_rouge = {
        "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
        "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
        "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0},
    }

    for rouge_type in avg_rouge.keys():
        avg_rouge[rouge_type]["f"] = np.mean(
            [score[rouge_type]["f"] for score in rouge_scores]
        )
        avg_rouge[rouge_type]["p"] = np.mean(
            [score[rouge_type]["p"] for score in rouge_scores]
        )
        avg_rouge[rouge_type]["r"] = np.mean(
            [score[rouge_type]["r"] for score in rouge_scores]
        )

    return avg_bleu, avg_rouge


def plot_scores(bleu_scores, rouge_scores, index):
    # 绘制BLEU分数分布
    plt.figure(figsize=(12, 6))
    sns.histplot(bleu_scores, bins=20, kde=True)
    plt.title("BLEU Score Distribution")
    plt.xlabel("BLEU Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{index}_bleu_score.png")

    # 绘制ROUGE分数分布
    for rouge_type in ["rouge-1", "rouge-2", "rouge-l"]:
        plt.figure(figsize=(12, 6))
        scores = [score[rouge_type]["f"] for score in rouge_scores]
        sns.histplot(scores, bins=20, kde=True)
        plt.title(f"{rouge_type.upper()} F-Score Distribution")
        plt.xlabel("F-Score")
        plt.ylabel("Frequency")
        plt.savefig(f"{index}_{rouge_type}_f_score.png")


def evaluate(
    dataset, model, device, epoch_num, batch_size=64, eval_stage="classification"
):
    """
    评估函数
    eval_stage:
        - 'classification': 只评估分类任务
        - 'generation': 只评估生成任务
        - 'joint': 联合评估
    """
    model.eval()

    # 初始化指标
    total_loss = 0
    total_cliloss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_generated_text = []
    all_golden_text = []

    # 确定是否需要生成评估
    need_generation = (eval_stage in ["generation", "joint"]) and (
        (epoch_num + 1) % 10 == 0
    )

    try:
        with torch.no_grad():
            dataset.load_batch_files(0)
            dataloader = GraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            for batch_idx, (
                g_premise,
                g_hypothesis,
                lexical_chain,
                hyp_emb,
                exp_text,
            ) in enumerate(dataloader):
                # 准备标签
                true_labels = [0, 1, 2] * len(hyp_emb)
                labels = torch.tensor(true_labels).to(device)

                # 处理图结构
                combined_graphs = [
                    merge_graphs(g_p, g_h, lc, rel_names_short)
                    for g_p, g_h, lc in zip(g_premise, g_hypothesis, lexical_chain)
                ]
                combined_graphs = dgl.batch(combined_graphs).to(device)
                hg_combined = model.encode_graph_combined(combined_graphs)

                # 处理嵌入
                hyp_tensor_embeddings = hyp_emb.to(device)

                # 根据评估阶段计算损失
                if eval_stage in ["classification", "joint"]:
                    triplet_loss = batch_triplet_loss_with_neutral(
                        hg_combined, hyp_tensor_embeddings
                    )
                else:
                    triplet_loss = torch.tensor(0.0).to(device)

                # 获取不同类型的嵌入
                embeddings = {
                    "positive": hyp_tensor_embeddings[:, 0, :],
                    "neutral": hyp_tensor_embeddings[:, 1, :],
                    "negative": hyp_tensor_embeddings[:, 2, :],
                }

                # 初始化损失和预测
                classification_loss = 0
                gen_loss = 0
                batch_predictions = []

                # 处理每种关系类型
                for idx, (rel_type, embedding) in enumerate(
                    [
                        ("entailment", embeddings["positive"]),
                        ("neutral", embeddings["neutral"]),
                        ("contradiction", embeddings["negative"]),
                    ]
                ):
                    target = torch.full(
                        (len(hyp_emb),), idx, dtype=torch.long, device=device
                    )

                    # 根据评估阶段决定是否需要生成文本
                    if eval_stage == "generation":
                        output = model(hg_combined, embedding, exp_text[idx])
                        gen_loss += output.get("gen_loss", 0)
                        if need_generation:
                            generated = model.generate_explanation(
                                hg_combined, embedding
                            )
                            all_generated_text.extend(generated)
                            all_golden_text.extend(exp_text[idx])

                    elif eval_stage == "classification":
                        output = model(hg_combined, embedding, None)  # 不进行生成
                        cls_loss = F.cross_entropy(output["logits"], target)
                        classification_loss += cls_loss
                        predicted = torch.argmax(output["logits"], dim=-1)
                        batch_predictions.append(predicted)

                    else:  # joint
                        output = model(hg_combined, embedding, exp_text[idx])
                        cls_loss = F.cross_entropy(output["logits"], target)
                        classification_loss += cls_loss
                        gen_loss += output.get("gen_loss", 0)
                        predicted = torch.argmax(output["logits"], dim=-1)
                        batch_predictions.append(predicted)

                        if need_generation:
                            generated = model.generate_explanation(
                                hg_combined, embedding
                            )
                            all_generated_text.extend(generated)
                            all_golden_text.extend(exp_text[idx])

                # 根据评估阶段计算总损失
                if eval_stage == "classification":
                    loss_cli = triplet_loss + classification_loss
                    total_loss += loss_cli.item()
                    total_cliloss += loss_cli.item()
                elif eval_stage == "generation":
                    total_loss += gen_loss.item()
                else:  # joint
                    loss_cli = triplet_loss + classification_loss
                    total_loss += (loss_cli + gen_loss * 0.5).item()
                    total_cliloss += loss_cli.item()

                # 只在分类和联合评估时更新分类指标
                if eval_stage in ["classification", "joint"] and batch_predictions:
                    stacked_preds = torch.stack(batch_predictions)
                    batch_preds = stacked_preds.T.flatten()
                    total_correct += (batch_preds == labels).sum().item()
                    total_samples += labels.size(0)
                    all_predictions.extend(batch_preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                # 定期清理GPU内存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        # 计算最终指标
        dataset_size = dataset.file_offsets[0][1]

        if eval_stage == "generation":
            avg_loss = total_loss / dataset_size
            ave_cliloss = 0
            accuracy = 0
            f1 = 0
        else:
            ave_cliloss = total_cliloss / dataset_size
            avg_loss = total_loss / dataset_size
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            f1 = (
                f1_score(all_labels, all_predictions, average="weighted")
                if all_predictions
                else 0
            )

        # 如果需要生成评估
        if need_generation and all_generated_text:
            bleu_score, rouge_score = cal_bleu_rouge(
                all_generated_text, all_golden_text
            )

            # 保存生成结果
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_texts_to_json(
                all_generated_text,
                all_golden_text,
                f"generated_output_{eval_stage}_{current_time}.json",
            )
            logging.info(
                f"Generated {len(all_generated_text)} texts in {eval_stage} stage."
            )

            return ave_cliloss, avg_loss, accuracy, f1, bleu_score, rouge_score

        return ave_cliloss, avg_loss, accuracy, f1, None, None

    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return None


def collate_fn(batch):
    g_premises, g_hypotheses, lexical_chains, hyp_emb_threes, exp_text_threes = zip(
        *batch
    )
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

    # 确保 exp_text_threes 是一个可迭代对象
    exp_text_threes = list(exp_text_threes)

    # 提取 exp_text_threes 中的子列表的元素
    list_0, list_1, list_2 = zip(*exp_text_threes)

    return (
        list(g_premises),
        list(g_hypotheses),
        list(lexical_chains),
        hyp_emb_threes,
        (list(list_0), list(list_1), list(list_2)),  # 返回三个列表
    )


def train(
    dataset,
    model,
    dev_dataset,
    epochs=10,
    lr=2e-5,
    save_path="best_model.pth",
    batch_size=64,
    device=device,
    grad_clip=1.0,
    warmup_ratio=0.1,
    training_stage="classification",  # 新增参数，用于控制训练阶段
    min_lr_ratio=0.1,  # 最小学习率比例
):
    """
    分阶段训练函数
    training_stage:
        - 'classification': 只训练分类任务
        - 'generation': 只训练生成任务
        - 'joint': 联合训练
    """
    model.to(device)

    # 根据训练阶段设置参数组和学习率
    if training_stage == "classification":
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "t5" not in n],
                "lr": lr,
                "min_lr": lr * min_lr_ratio,  # 添加最小学习率
            }
        ]
        for name, param in model.named_parameters():
            if "t5" in name:
                param.requires_grad = False

    elif training_stage == "generation":
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "t5" in n],
                "lr": lr,
                "min_lr": lr * min_lr_ratio,
            },
            {
                "params": [p for n, p in model.named_parameters() if "t5" not in n],
                "lr": lr * 0.1,
                "min_lr": lr * 0.1 * min_lr_ratio,
            },
        ]
        for name, param in model.named_parameters():
            param.requires_grad = True

    else:  # joint training
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "t5" in n],
                "lr": lr * 0.1,
                "min_lr": lr * 0.1 * min_lr_ratio,
            },
            {
                "params": [p for n, p in model.named_parameters() if "t5" not in n],
                "lr": lr,
                "min_lr": lr * min_lr_ratio,
            },
        ]
        for name, param in model.named_parameters():
            param.requires_grad = True
            param.requires_grad = True

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    # 计算总步数用于warmup
    total_steps = dataset.file_offsets[0][1] // batch_size * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,  # 每次将学习率降低到原来的一半
        patience=2,  # 等待2个epoch无改善再降低学习率
        min_lr=[group["min_lr"] for group in optimizer_grouped_parameters],
        verbose=True,
    )

    best_dev_f1 = 0.0
    gen_loss_weight = torch.nn.Parameter(torch.tensor(0.1)).to(device)

    dataset.load_batch_files(0)
    dataloader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"triplet": 0, "classification": 0, "generation": 0, "total": 0}

        for index, (
            g_premise,
            g_premise2,
            lexical_chain,
            hyp_emb,
            exp_text,
        ) in enumerate(dataloader):
            optimizer.zero_grad()

            with autocast():
                # 图处理部分保持不变
                combined_graphs = [
                    merge_graphs(g_p1, g_p2, lc, rel_names_short)
                    for g_p1, g_p2, lc in zip(g_premise, g_premise2, lexical_chain)
                ]
                combined_graphs = dgl.batch(combined_graphs).to(device)
                hg_combined = model.encode_graph_combined(combined_graphs)

                # 假设处理
                hyp_tensor_embeddings = hyp_emb.to(device)
                positive = hyp_tensor_embeddings[:, 0, :]
                neutral = hyp_tensor_embeddings[:, 1, :]
                negative = hyp_tensor_embeddings[:, 2, :]

                # 根据训练阶段计算损失
                if training_stage == "classification":
                    # 只计算分类相关损失
                    triplet_loss = batch_triplet_loss_with_neutral(
                        hg_combined, hyp_tensor_embeddings
                    )
                    classification_losses = []

                    for embeddings, targets, _, _ in [
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
                            torch.full(
                                (len(hyp_emb),), 2, dtype=torch.long, device=device
                            ),
                            None,
                            "contradiction",
                        ),
                    ]:
                        outputs = model(hg_combined, embeddings, None)  # 不进行生成
                        cls_loss = F.cross_entropy(outputs["logits"], targets)
                        classification_losses.append(cls_loss)

                    classification_loss = sum(classification_losses)
                    total_loss = triplet_loss + classification_loss
                    generation_loss = torch.tensor(0.0).to(device)

                elif training_stage == "generation":
                    # 只计算生成损失
                    generation_losses = []
                    for embeddings, _, texts, _ in [
                        (positive, None, exp_text[0], "entailment"),
                        (neutral, None, exp_text[1], "neutral"),
                        (negative, None, exp_text[2], "contradiction"),
                    ]:
                        outputs = model(hg_combined, embeddings, texts)
                        if "gen_loss" in outputs:
                            gen_loss = outputs["gen_loss"]
                            # if torch.abs(gen_loss) > 10:
                            #     gen_loss = 10 * (gen_loss / torch.abs(gen_loss))
                            generation_losses.append(gen_loss)

                    generation_loss = (
                        sum(generation_losses)
                        if generation_losses
                        else torch.tensor(0.0).to(device)
                    )
                    total_loss = generation_loss
                    triplet_loss = classification_loss = torch.tensor(0.0).to(device)

                else:  # joint training
                    # 完整的损失计算（原始代码的损失计算部分）
                    triplet_loss = batch_triplet_loss_with_neutral(
                        hg_combined, hyp_tensor_embeddings
                    )
                    classification_losses = []
                    generation_losses = []
                    for embeddings, targets, texts, _ in [
                        (
                            positive,
                            torch.zeros(len(hyp_emb), dtype=torch.long, device=device),
                            exp_text[0],
                            "entailment",
                        ),
                        (
                            neutral,
                            torch.ones(len(hyp_emb), dtype=torch.long, device=device),
                            exp_text[1],
                            "neutral",
                        ),
                        (
                            negative,
                            torch.full(
                                (len(hyp_emb),), 2, dtype=torch.long, device=device
                            ),
                            exp_text[2],
                            "contradiction",
                        ),
                    ]:
                        outputs = model(hg_combined, embeddings, texts)
                        cls_loss = F.cross_entropy(outputs["logits"], targets)
                        classification_losses.append(cls_loss)

                        if "gen_loss" in outputs:
                            gen_loss = outputs["gen_loss"]
                            if torch.abs(gen_loss) > 10:
                                gen_loss = 10 * (gen_loss / torch.abs(gen_loss))
                            generation_losses.append(gen_loss)

                    classification_loss = sum(classification_losses)
                    generation_loss = (
                        sum(generation_losses)
                        if generation_losses
                        else torch.tensor(0.0).to(device)
                    )

                    current_step = epoch * len(dataloader) + index
                    warmup_factor = min(1.0, current_step / warmup_steps)
                    gen_weight = warmup_factor * torch.sigmoid(gen_loss_weight)
                    total_loss = (
                        triplet_loss
                        + classification_loss
                        + generation_loss * gen_weight
                    )
                torch.cuda.empty_cache()

                # 记录损失
                epoch_losses["triplet"] += triplet_loss.item()
                epoch_losses["classification"] += classification_loss.item()
                epoch_losses["generation"] += generation_loss.item()
                epoch_losses["total"] += total_loss.item()

            # 反向传播和优化
            total_loss.backward()

            # 梯度裁剪
            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    if "t5" in name:
                        torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(param, grad_clip)

            optimizer.step()
            # scheduler.step()

        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} - Stage: {training_stage}")
        for loss_type, loss_value in epoch_losses.items():
            avg_loss = loss_value / len(dataloader)
            print(f"Average {loss_type} loss: {avg_loss:.4f}")

        # 评估
        dev_cliloss, dev_loss, dev_accuracy, dev_f1, bleu, rouge = evaluate(
            dev_dataset,
            model,
            device,
            epoch,
            batch_size=batch_size,
            eval_stage=training_stage,
        )

        if bleu is not None and rouge is not None:
            avg_bleu, avg_rouge = calculate_average_scores(bleu, rouge)
            logging.info(
                f"Epoch {epoch+1}/{epochs} - Stage: {training_stage}, "
                f"average bleu: {avg_bleu}, average rouge: {avg_rouge}"
            )

        logging.info(
            f"Epoch {epoch+1}/{epochs} - Stage: {training_stage}, "
            f"Validation Loss: {dev_loss}, Validation Cliloss: {dev_cliloss}, "
            f"Validation Accuracy: {dev_accuracy}, dev_f1 Score: {dev_f1}"
        )

        # 保存模型
        if training_stage == "generation":
            save_condition = bleu is not None and rouge is not None and avg_bleu > 0.01
        else:
            save_condition = (dev_f1 > best_dev_f1 + 0.03) and dev_f1 > 0.6

        if save_condition:
            best_dev_f1 = dev_f1
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_save_path = (
                f"model_{training_stage}_{save_path}_epoch{epoch+1}_{current_time}.pth"
            )
            torch.save(model.state_dict(), full_save_path)
            logging.info(f"Model saved at epoch {epoch+1}: {full_save_path}")

        scheduler.step(dev_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr}")

    avg_loss = epoch_losses["total"] / epochs
    avg_cliloss = epoch_losses["classification"] / epochs
    logging.info(
        f"Stage {training_stage} completed - "
        f"Average total loss: {avg_loss}, Average classification loss: {avg_cliloss}"
    )
    if full_save_path:

        return full_save_path
    else:
        return None


def test(dataset, model, device, batch_size=128):
    model.eval()
    model.to(device)

    # 初始化指标
    metrics = {"total_correct": 0, "total_samples": 0}

    # 收集预测和生成结果
    results = {
        "predictions": [],
        "labels": [],
        "generated_texts": [],
        "golden_texts": [],
    }

    try:
        with torch.no_grad():
            dataset.load_batch_files(0)
            dataloader = GraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            for batch_idx, (
                g_premise,
                g_hypothesis,
                lexical_chain,
                hyp_emb,
                exp_text,
            ) in enumerate(dataloader):
                # 准备标签
                true_labels = [0, 1, 2] * len(hyp_emb)
                labels = torch.tensor(true_labels).to(device)

                # 处理图结构
                combined_graphs = [
                    merge_graphs(g_p, g_h, lc, rel_names_short)
                    for g_p, g_h, lc in zip(g_premise, g_hypothesis, lexical_chain)
                ]
                combined_graphs = dgl.batch(combined_graphs).to(device)
                hg_combined = model.encode_graph_combined(combined_graphs)

                # 处理嵌入
                hyp_tensor_embeddings = hyp_emb.to(device)
                embeddings = {
                    "positive": hyp_tensor_embeddings[:, 0, :],  # 蕴含
                    "neutral": hyp_tensor_embeddings[:, 1, :],  # 中立
                    "negative": hyp_tensor_embeddings[:, 2, :],  # 矛盾
                }

                # 存储每种关系的预测和生成结果
                batch_predictions = []
                batch_generated = []

                # 处理每种关系类型
                for idx, (rel_type, embedding) in enumerate(
                    [
                        ("entailment", embeddings["positive"]),
                        ("neutral", embeddings["neutral"]),
                        ("contradiction", embeddings["negative"]),
                    ]
                ):
                    # 模型推理
                    output = model(hg_combined, embedding, exp_text[idx])

                    # 收集预测和生成结果
                    batch_predictions.append(torch.argmax(output["logits"], dim=-1))
                    batch_generated.extend(
                        model.generate_explanation(hg_combined, embedding)
                    )
                    results["golden_texts"].extend(exp_text[idx])

                # 处理预测结果
                stacked_preds = torch.stack(batch_predictions)
                batch_preds = stacked_preds.T.flatten()

                # 更新指标
                metrics["total_correct"] += (batch_preds == labels).sum().item()
                metrics["total_samples"] += labels.size(0)

                # 收集结果
                results["predictions"].extend(batch_preds.cpu().numpy())
                results["labels"].extend(labels.cpu().numpy())
                results["generated_texts"].extend(batch_generated)

                # 定期清理GPU内存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        # 计算最终指标
        accuracy = metrics["total_correct"] / metrics["total_samples"]
        f1 = f1_score(results["labels"], results["predictions"], average="weighted")

        # 计算BLEU和ROUGE分数
        bleu, rouge = cal_bleu_rouge(
            results["generated_texts"], results["golden_texts"]
        )
        average_bleu, average_rouge = calculate_average_scores(bleu, rouge)

        # 保存生成结果
        save_texts_to_json(
            results["generated_texts"], results["golden_texts"], "generated_output.json"
        )

        # 记录结果
        logging.info(
            f"Test Results:\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
            f"Average BLEU: {average_bleu:.4f}\n"
            f"Average ROUGE: {average_rouge:.4f}"
        )

        return accuracy, f1, bleu, rouge

    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        traceback.print_exc()
        return None


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def load_pretrained_model(model_path, model, training_stage="generation"):
    """
    加载预训练模型并优化内存使用
    """
    # 首先清理GPU缓存
    torch.cuda.empty_cache()

    # 加载检查点
    checkpoint = torch.load(model_path, map_location="cpu")  # 先加载到CPU

    # 如果只训练生成任务，冻结其他部分参数
    if training_stage == "generation":
        # 冻结RGAT和分类器参数
        for name, param in model.named_parameters():
            if "t5" not in name and "feature_proj" not in name:
                param.requires_grad = False

        # 优化T5配置
        model.t5.config.use_cache = False
        model.t5.gradient_checkpointing_enable()

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


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
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_pairs_for_exp/train"
    )
    evidence_train_text_output_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/train/hyp/evidence_text.json"
    )

    logging.info("Processing train data")
    train_dataset = RSTDataset(
        train_rst_path,
        model_output_path,
        model_output_emb_path,
        evidence_train_text_output_path,
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
    dev_pair_graph = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_pairs_for_exp/dev"
    )
    evidence_dev_text_output_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/dev/hyp/evidence_text.json"
    )
    dev_dataset = RSTDataset(
        dev_rst_path,
        dev_model_output_path,
        dev_model_output_emb_path,
        evidence_dev_text_output_path,
        dev_lexical_chains_path,
        dev_embedding_file,
        batch_file_size,
        save_dir=dev_pair_graph,
    )

    logging.info("Processing test data")

    test_rst_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/test/test1/new_rst_result.jsonl"
    )
    test_model_output_path = (
        r"/mnt/nlp/yuanmengying/nli_data_generate/test_all_generated_hypothesis.json"
    )
    test_model_output_emb_path = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/test/hyp/hypothesis_embeddings.npz"
    test_lexical_chains_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_infos/test"
    )
    test_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/cross_document/test/pre"
    test_pair_graph = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/graph_pairs_for_exp/test"
    )
    evidence_test_text_output_path = (
        r"/mnt/nlp/yuanmengying/ymy/data/cross_document/test/hyp/evidence_text.json"
    )
    test_dataset = RSTDataset(
        test_rst_path,
        test_model_output_path,
        test_model_output_emb_path,
        evidence_test_text_output_path,
        test_lexical_chains_path,
        test_embedding_file,
        batch_file_size,
        save_dir=test_pair_graph,
    )

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

    model = ExplainableHeteroClassifier(
        in_dim=1024,
        hidden_dim=1024,
        n_classes=3,
        rel_names=rel_names_short,
        decoder_name=decode_name,
    )

    # # 训练模型

    # 1. 先训练分类模型
    # cli_model_path = train(
    #     train_dataset,
    #     model,
    #     dev_dataset,
    #     epochs=20,
    #     lr=0.001,
    #     training_stage="classification",
    # )
    # 2. 加载最佳分类模型后训练生成

    # model.load_state_dict(torch.load(cli_model_path))
    # model_path = "/mnt/nlp/yuanmengying/ymy/new_data_process/model_classification_best_model.pth_epoch19_20241031_165928.pth"
    # model = load_pretrained_model(model_path, model, training_stage="generation")
    full_save_path = "/mnt/nlp/yuanmengying/ymy/new_data_process/model_classification_best_model.pth_epoch19_20241031_165928.pth"
    model.load_state_dict(torch.load(full_save_path), strict=False)
    gen_model_path = train(
        train_dataset,
        model,
        dev_dataset,
        epochs=50,
        lr=1e-7,
        batch_size=16,
        training_stage="generation",
    )
    # 3. 最后进行联合训练

    model.load_state_dict(gen_model_path)

    train(train_dataset, model, dev_dataset, epochs=10, training_stage="joint")

    # full_save_path = "/mnt/nlp/yuanmengying/ymy/new_data_process/model_classification_best_model.pth_epoch19_20241031_165928.pth"  # 5+20+30的
    # model.load_state_dict(torch.load(full_save_path))
    # train(
    #     train_dataset,
    #     model,
    #     dev_dataset,
    #     epochs=30,
    #     lr=1e-5,
    #     device=device,
    #     batch_size=16,
    # )

    accuracy, f1, bleu, rouge = test(test_dataset, model, device, batch_size=32)

    # plot_scores(bleu, rouge, 1234)
