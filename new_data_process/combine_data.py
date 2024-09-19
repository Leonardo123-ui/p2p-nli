import os
import pickle
import torch
import re


def get_npz_files(directory, type):
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(type):
                npz_files.append(os.path.join(root, file))
    return npz_files


# 定义一个函数来提取文件名中的数字
def extract_number(filename):
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else float("inf")


def merge_files_to_single_pkl(pkl_file_dir, npz_file_dir, output_pkl, output_npz):
    # 按文件名中的数字顺序对 .pkl 文件排序
    pkl_filenames = get_npz_files(pkl_file_dir, ".pkl")

    # 根据文件名中的数字进行排序
    sorted_pkl_paths = sorted(pkl_filenames, key=extract_number)
    npz_filenames = get_npz_files(npz_file_dir, ".npz")
    # 按文件名中的数字顺序对 .npz 文件排序
    sorted_npz_paths = sorted(npz_filenames, key=extract_number)

    # # 合并 .pkl 文件中的数据
    # all_pkl_data = []
    # for file_path in sorted_pkl_paths:
    #     with open(file_path, "rb") as f:
    #         data = pickle.load(f)
    #         all_pkl_data.extend(data)  # 合并所有 .pkl 数据
    #     print(f"Loaded and merged .pkl data from {file_path}")
    # print(f"all_pkl_data length: ", len(all_pkl_data))
    # # 将合并后的 .pkl 数据保存到一个新文件
    # with open(output_pkl, "wb") as f:
    #     pickle.dump(all_pkl_data, f)
    # print(f"All .pkl data saved to {output_pkl}")

    # 合并 .npz 文件中的数据
    all_npz_data = []
    for file_path in sorted_npz_paths:
        data = torch.load(file_path)
        all_npz_data.extend(data)  # 合并所有 .npz 数据
        print(f"Loaded and merged .npz data from {file_path}")
    print(f"all_npz_data length: ", len(all_npz_data))
    # 将合并后的 .npz 数据保存到一个新文件
    torch.save(all_npz_data, output_npz)
    print(f"All .npz data saved to {output_npz}")


# # 示例用法，分别合并 .pkl 和 .npz 文件
# pkl_files = ['file1.pkl', 'file2.pkl', 'file3.pkl']  # 替换为实际 .pkl 文件路径
# npz_files = ['file1.npz', 'file2.npz', 'file3.npz']  # 替换为实际 .npz 文件路径

# output_pkl = 'merged_lexical_chains.pkl'
# output_npz = 'merged_embeddings.npz'

train_directory_emb = "/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/train"
train_directory_lexical = "/mnt/nlp/yuanmengying/ymy/data/graph_infos/train"
train_output_lexical = (
    "/mnt/nlp/yuanmengying/ymy/data/merged_processed/train/merged_lexical_chains.pkl"
)
train_output_emb = (
    "/mnt/nlp/yuanmengying/ymy/data/merged_processed/train/merged_embeddings.npz"
)

dev_directory_emb = "/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/dev"
dev_directory_lexical = "/mnt/nlp/yuanmengying/ymy/data/graph_infos/dev"
dev_output_lexical = (
    "/mnt/nlp/yuanmengying/ymy/data/merged_processed/dev/merged_lexical_chains.pkl"
)
dev_output_emb = (
    "/mnt/nlp/yuanmengying/ymy/data/merged_processed/dev/merged_embeddings.npz"
)

test_directory_emb = "/mnt/nlp/yuanmengying/ymy/data/processed_docnli_origin/test"
test_directory_lexical = "/mnt/nlp/yuanmengying/ymy/data/graph_infos/test"
test_output_lexical = (
    "/mnt/nlp/yuanmengying/ymy/data/merged_processed/test/merged_lexical_chains.pkl"
)
test_output_emb = (
    "/mnt/nlp/yuanmengying/ymy/data/merged_processed/test/merged_embeddings.npz"
)

# merge_files_to_single_pkl(
#     train_directory_lexical, train_directory_emb, train_output_lexical, train_output_emb
# )
# merge_files_to_single_pkl(
#     dev_directory_lexical, dev_directory_emb, dev_output_lexical, dev_output_emb
# )
merge_files_to_single_pkl(
    test_directory_lexical, test_directory_emb, test_output_lexical, test_output_emb
)
