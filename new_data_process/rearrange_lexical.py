import pickle
import os
import re

file_paths = []
lexical_chains = []
lexical_chains_path = "/mnt/nlp/yuanmengying/ymy/data/graph_infos/train"
new_lexical_chains_path = "/mnt/nlp/yuanmengying/ymy/data/graph_infos_2/train"
for filename in os.listdir(lexical_chains_path):
    if filename.endswith(".pkl"):
        file_path = os.path.join(lexical_chains_path, filename)
        file_paths.append(file_path)
print(len(file_paths))
file_paths.sort(key=lambda x: int(re.search(r"\d+", x).group()))
for file_path in file_paths:
    lexical_chain = pickle.load(open(file_path, "rb"))
    lexical_chains.extend(lexical_chain)
print("all len:", len(lexical_chains))
batches = []
if lexical_chains:
    batches.append(lexical_chains[0])  # 保存第一个元素
    file = os.path.join(new_lexical_chains_path, "lexical_0.pkl")
    pickle.dump(batches, open(file, "wb"))
    batches = []
# 然后按照5000的步长保存其余元素
count = 1
# 然后按照5000的步长保存其余元素
for i in range(1, len(lexical_chains), 5000):
    batch = lexical_chains[i : i + 5000]  # 保存每5000个元素
    file = os.path.join(new_lexical_chains_path, f"lexical_{count}.pkl")
    print("batch len", len(batch))  # 打印当前 batch 的长度

    with open(file, "wb") as f:
        pickle.dump(batch, f)  # 只保存当前 batch

    count += 1  # 增加计数器
print("done")
