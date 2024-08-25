import json


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


data = load_json("/mnt/nlp/yuanmengying/ymy/data/DocNLI_dataset/train.json")

for i in range(len(data) - 920949):
    pre = data[920949 + i]["premise"]
    hyp = data[920949 + i]["hypothesis"]
    if len(pre) < 10:
        print(pre, i + 920949, "pre")
    if len(hyp) < 10:
        print(hyp, i + 920949, "hyp")
print("end")
# import os


# def rename_npz_to_pkl(directory_path):
#     # 遍历目录中的所有文件
#     for filename in os.listdir(directory_path):
#         # 检查文件是否以 .npz 结尾
#         if filename.endswith(".npy"):
#             # 构建完整的文件路径
#             old_file_path = os.path.join(directory_path, filename)
#             # 构建新的文件名和路径
#             new_filename = filename[:-4] + ".pkl"
#             new_file_path = os.path.join(directory_path, new_filename)
#             # 重命名文件
#             os.rename(old_file_path, new_file_path)
#             print(f"Renamed: {old_file_path} to {new_file_path}")


# # 使用示例
# directory_path = "/mnt/nlp/yuanmengying/ymy/data/graph_infos/train"
# rename_npz_to_pkl(directory_path)
