import dgl
import torch
from datetime import datetime
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"DGL version: {dgl.__version__}")
import logging
from build_base_graph import merge_graphs
from ymy.new_data_process.data_loader_old import create_dataloader
from build_base_graph import HeteroClassifier

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate(dataloader, model, device=device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for g_premises, g_hypotheses, lexical_chains, labels in dataloader:
            combined_graphs = [
                merge_graphs(g_p, g_h, lc)
                for g_p, g_h, lc in zip(g_premises, g_hypotheses, lexical_chains)
            ]
            combined_graphs = dgl.batch(combined_graphs).to(device)
            labels = labels.to(device)
            outputs = model(combined_graphs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 收集所有预测和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    f1 = f1_score(
        all_labels, all_predictions, average="weighted"
    )  # 使用加权平均计算 F1 值

    logging.info(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}, F1 Score: {f1}")

    return avg_loss, accuracy, f1


def train(
    dataloader,
    model,
    dev_dataloader,
    epochs=10,
    lr=0.01,
    save_path="best_model.pth",
    device=device,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_dev_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for g_premises, g_hypotheses, lexical_chains, labels in dataloader:
            optimizer.zero_grad()
            combined_graphs = [
                merge_graphs(g_p, g_h, lc)
                for g_p, g_h, lc in zip(g_premises, g_hypotheses, lexical_chains)
            ]
            combined_graphs = dgl.batch(combined_graphs).to(device)
            labels = labels.to(device)
            outputs = model(combined_graphs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

        dev_loss, dev_accuracy, f1 = evaluate(dev_dataloader, model, device)
        logging.info(
            f"Epoch {epoch+1}/{epochs}, Validation Loss: {dev_loss}, Validation Accuracy: {dev_accuracy}, f1 score: {f1}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_save_path = f"{save_path}_epoch{epoch+1}_{current_time}.pth"
            # torch.save(model.state_dict(), save_path)
            # 保存模型状态字典
            torch.save(model.state_dict(), full_save_path)
            # 记录日志信息
            logging.info(f"Model saved at epoch {epoch+1} with path: {full_save_path}")


def test(dataloader, model, device=device):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for g_premises, g_hypotheses, lexical_chains, labels in dataloader:
            combined_graphs = [
                merge_graphs(g_p, g_h, lc)
                for g_p, g_h, lc in zip(g_premises, g_hypotheses, lexical_chains)
            ]
            combined_graphs = dgl.batch(combined_graphs).to(device)
            labels = labels.to(device)
            outputs = model(combined_graphs)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            # 收集所有预测和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = total_correct / total_samples
    f1 = f1_score(
        all_labels, all_predictions, average="weighted"
    )  # 使用加权平均计算 F1 值
    logging.info(f"Test Accuracy: {accuracy}, F1 Score: {f1}")

    return accuracy, f1


if __name__ == "__main__":
    # 加载数据和模型

    batch_size = 100
    train_save_dir = "/mnt/nlp/yuanmengying/ymy/data/graph_pairs/train/graph_pairs.bin"
    dev_save_dir = "/mnt/nlp/yuanmengying/ymy/data/graph_pairs/dev/graph_pairs.bin"
    test_save_dir = "/mnt/nlp/yuanmengying/ymy/data/graph_pairs/test/graph_pairs.bin"

    train_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/train/new_rst_result.jsonl"
    train_lexical_chains_path = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos/train"
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
    train_dataloader = create_dataloader(
        train_rst_path,
        train_lexical_chains_path,
        train_embedding_file,
        train_save_dir,
        batch_size,
        subset_ratio=1 / 1000,
    )

    logging.info("Processing dev data")

    dev_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/dev/new_rst_result.jsonl"
    dev_lexical_chains_path = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos/dev"
    dev_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/dev"
    # dev_rst_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev/new_rst_result.jsonl"
    # )
    # dev_lexical_chains_path = (
    #     r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev/matrixes.pkl"
    # )
    # dev_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/temp_data_small_dev"
    dev_dataloader = create_dataloader(
        dev_rst_path,
        dev_lexical_chains_path,
        dev_embedding_file,
        dev_save_dir,
        batch_size,
        subset_ratio=1 / 1000,
    )
    # logging.info("Processing test data")
    # test_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/test/new_rst_result.jsonl"
    # test_lexical_chains_path = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos/test"
    # test_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/test"
    # test_dataloader = create_dataloader(
    #     test_rst_path,
    #     test_lexical_chains_path,
    #     test_embedding_file,
    #     test_save_dir,
    #     batch_size,
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
    train(train_dataloader, model, dev_dataloader, epochs=50, device=device)
