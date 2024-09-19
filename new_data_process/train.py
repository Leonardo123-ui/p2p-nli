import dgl
import torch
from datetime import datetime
from sklearn.metrics import f1_score
from build_base_graph import build_graph
from dgl.dataloading import GraphDataLoader
from data_loader import RSTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"DGL version: {dgl.__version__}")
import logging
from build_base_graph import merge_graphs

# from data_loader import create_dataloader
from build_base_graph import HeteroClassifier

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

            for g_premise, g_hypothesis, lexical_chain, labels in dataloader:
                # 将数据移动到 GPU
                labels = labels.to(device)

                # 合并图结构
                combined_graphs = [
                    merge_graphs(g_p, g_h, lc)
                    for g_p, g_h, lc in zip(g_premise, g_hypothesis, lexical_chain)
                ]
                combined_graphs = dgl.batch(combined_graphs).to(device)

                # 前向传播
                outputs = model(combined_graphs)
                loss = criterion(outputs, labels)

                # 记录损失
                total_loss += loss.item()

                # 预测标签
                _, predicted = torch.max(outputs, 1)

                # 记录正确的预测数量
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # 收集所有预测和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / dataset.total_batches
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    logging.info(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}, F1 Score: {f1}")
    return avg_loss, accuracy, f1


def extract_node_features(embeddings_data, idx, prefix):
    node_features = {}
    for item in embeddings_data[idx][prefix]:
        node_id, embedding = item
        node_features[node_id] = embedding
    return node_features


def collate_fn(batch):
    g_premises, g_hypotheses, lexical_chains, labels = zip(*batch)
    return (
        list(g_premises),
        list(g_hypotheses),
        list(lexical_chains),
        torch.tensor(labels),
    )


def train(
    dataset,
    model,
    dev_dataset,
    epochs=10,
    lr=0.01,
    save_path="best_model.pth",
    batch_size=256,  # 每次模型训练输入的batch大小
    device=device,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_dev_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # 遍历文件批次
        for batch_num in range(dataset.total_batches):
            print(f"Loading batch {batch_num + 1}/{dataset.total_batches}...")

            # 加载当前文件批次并构建图
            dataset.load_batch_files(batch_num)

            # 创建 DataLoader 对当前文件批次的数据进行迭代
            dataloader = GraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,  # 每次打乱顺序
                collate_fn=collate_fn,
            )

            for g_premise, g_hypothesis, lexical_chain, labels in dataloader:
                optimizer.zero_grad()

                # 将数据移动到 GPU
                # g_premise = [g.to(device) for g in g_premise]
                # g_hypothesis = [g.to(device) for g in g_hypothesis]
                # lexical_chain = [torch.tensor(lc).to(device) for lc in lexical_chain]
                labels = labels.to(device)

                # 这里 merge_graphs 函数将 premise 和 hypothesis 图合并
                combined_graphs = [
                    merge_graphs(g_p, g_h, lc)
                    for g_p, g_h, lc in zip(g_premise, g_hypothesis, lexical_chain)
                ]
                combined_graphs = dgl.batch(combined_graphs).to(device)

                outputs = model(combined_graphs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / dataset.total_batches
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

        dev_loss, dev_accuracy, f1 = evaluate(dev_dataset, model, device)
        logging.info(
            f"Epoch {epoch+1}/{epochs}, Validation Loss: {dev_loss}, Validation Accuracy: {dev_accuracy}, F1 Score: {f1}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_save_path = f"{save_path}_epoch{epoch+1}_{current_time}.pth"
            torch.save(model.state_dict(), full_save_path)
            logging.info(f"Model saved at epoch {epoch+1} with path: {full_save_path}")


def test(dataset, model, device, batch_size=128):
    model.eval()  # 切换到评估模式
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        # 遍历文件批次
        for batch_num in range(dataset.total_batches):
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

            for g_premise, g_hypothesis, lexical_chain, labels in dataloader:
                # 将数据移动到 GPU
                labels = labels.to(device)

                # 合并图结构
                combined_graphs = [
                    merge_graphs(g_p, g_h, lc)
                    for g_p, g_h, lc in zip(g_premise, g_hypothesis, lexical_chain)
                ]
                combined_graphs = dgl.batch(combined_graphs).to(device)

                # 前向传播
                outputs = model(combined_graphs)

                # 预测标签
                _, predicted = torch.max(outputs, 1)

                # 记录正确的预测数量
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # 收集所有预测和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # 计算准确率和F1
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    logging.info(f"Test Accuracy: {accuracy}, F1 Score: {f1}")
    return accuracy, f1


if __name__ == "__main__":
    # 加载数据和模型

    batch_file_size = 1

    train_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/train/new_rst_result.jsonl"
    train_lexical_chains_path = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos_2/train"
    train_embedding_file = (
        r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/train"
    )
    # train_rst_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev/new_rst_result.jsonl"
    # )
    # train_lexical_chains_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev/matrixes.pkl"
    # )
    # train_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev"
    logging.info("Processing train data")
    #!!!!
    # train_dataloader = create_dataloader(
    #     train_rst_path,
    #     train_lexical_chains_path,
    #     train_embedding_file,
    #     batch_file_size,
    # )
    #!!!!
    train_dataset = RSTDataset(
        train_rst_path, train_lexical_chains_path, train_embedding_file, batch_file_size
    )
    logging.info("Processing dev data")

    dev_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/dev/new_rst_result.jsonl"
    dev_lexical_chains_path = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos_2/dev"
    dev_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/dev"
    # dev_rst_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev/new_rst_result.jsonl"
    # )
    # dev_lexical_chains_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev/matrixes.pkl"
    # )
    # dev_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev"
    #!!!!
    # dev_dataloader = create_dataloader(
    #     dev_rst_path,
    #     dev_lexical_chains_path,
    #     dev_embedding_file,
    #     batch_file_size,
    # )
    #!!!!
    dev_dataset = RSTDataset(
        dev_rst_path, dev_lexical_chains_path, dev_embedding_file, batch_file_size
    )
    # logging.info("Processing test data")
    # test_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/test/new_rst_result.jsonl"
    # test_lexical_chains_path = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos_2/test"
    # test_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/test"
    # test_dataloader = create_dataloader(
    #     test_rst_path,
    #     test_lexical_chains_path,
    #     test_embedding_file,
    #     batch_file_size,
    # )
    num_rels = 20
    rel_names = [
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
        "span",
        "lexical",
    ]

    model = HeteroClassifier(
        in_dim=1024, hidden_dim=256, n_classes=2, rel_names=rel_names
    )

    # 训练模型
    train(train_dataset, model, dev_dataset, epochs=2, device=device)
