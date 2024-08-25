import dgl
import torch
from build_base_graph import merge_graphs
from data_loader import create_dataloader
from build_base_graph import HeteroClassifier


def train(
    dataloader, model, dev_dataloader, epochs=10, lr=0.01, save_path="best_model.pth"
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_dev_loss = 1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for g_premises, g_hypotheses, lexical_chains, labels in dataloader:
            optimizer.zero_grad()
            combined_graphs = [
                merge_graphs(g_p, g_h, lc)
                for g_p, g_h, lc in zip(g_premises, g_hypotheses, lexical_chains)
            ]
            combined_graphs = dgl.batch(combined_graphs)
            labels = labels.clone().detach().to(combined_graphs.device)
            outputs = model(combined_graphs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
        # 验证模型
        dev_loss, dev_accuracy = evaluate(dev_dataloader, model)
        print(
            f"Epoch {epoch+1}/{epochs}, Validation Loss: {dev_loss}, Validation Accuracy: {dev_accuracy}"
        )

        # 保存最优模型
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1}")


def evaluate(dataloader, model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for g_premises, g_hypotheses, lexical_chains, labels in dataloader:
            combined_graphs = [
                merge_graphs(g_p, g_h, lc)
                for g_p, g_h, lc in zip(g_premises, g_hypotheses, lexical_chains)
            ]
            combined_graphs = dgl.batch(combined_graphs)
            labels = labels.clone().detach().to(combined_graphs.device)
            outputs = model(combined_graphs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")

    return avg_loss, accuracy


def test(dataloader, model):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for g_premises, g_hypotheses, lexical_chains, labels in dataloader:
            combined_graphs = [
                merge_graphs(g_p, g_h, lc)
                for g_p, g_h, lc in zip(g_premises, g_hypotheses, lexical_chains)
            ]
            combined_graphs = dgl.batch(combined_graphs)
            labels = labels.clone().detach().to(combined_graphs.device)
            outputs = model(combined_graphs)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy}")

    return accuracy


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

    train_dataloader = create_dataloader(
        train_rst_path,
        train_lexical_chains_path,
        train_embedding_file,
        train_save_dir,
        batch_size,
    )
    print("process dev data")
    dev_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/dev/new_rst_result.jsonl"
    dev_lexical_chains_path = r"/mnt/nlp/yuanmengying/ymy/data/graph_infos/dev"
    dev_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/dev"
    dev_dataloader = create_dataloader(
        dev_rst_path,
        dev_lexical_chains_path,
        dev_embedding_file,
        dev_save_dir,
        batch_size,
    )

    # rst_results = []
    # with open(rst_path, "r") as file:
    #     for line in file:
    #         rst_dict = json.loads(line.strip())
    #         rst_results.append(rst_dict)

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
        "span",  # 可以考虑去掉，因为关s系不密切
        "lexical",  # 词汇链
    ]
    model = HeteroClassifier(
        in_dim=768, hidden_dim=256, n_classes=2, rel_names=rel_names
    )

    # 训练模型
    train(train_dataloader, model, dev_dataloader)
