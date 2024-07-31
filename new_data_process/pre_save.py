from data_loader import RSTDataset

train_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/temp_data_train/new_rst_result.jsonl"
train_lexical_chains_path = (
    r"/mnt/nlp/yuanmengying/ymy/data/temp_data_train/lexical_matrixes.pkl"
)
train_embedding_file = (
    r"/mnt/nlp/yuanmengying/ymy/data/temp_data_train/node_embeddings.npz"
)
train_data_loader = RSTDataset(
    train_rst_path, train_lexical_chains_path, train_embedding_file
)
dev_rst_path = r"/mnt/nlp/yuanmengying/ymy/data/temp_data_dev/new_rst_result.jsonl"
dev_lexical_chains_path = (
    r"/mnt/nlp/yuanmengying/ymy/data/temp_data_dev/lexical_matrixes.pkl"
)
dev_embedding_file = r"/mnt/nlp/yuanmengying/ymy/data/temp_data_dev/node_embeddings.npz"
dev_data_loader = RSTDataset(dev_rst_path, dev_lexical_chains_path, dev_embedding_file)
