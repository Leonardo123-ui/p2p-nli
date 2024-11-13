from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from data_loader_extract import RSTDataset
import logging
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)


@dataclass
class DataPaths:
    rst_path: str
    model_output_path: str
    model_output_emb_path: str
    evidence_text_output_path: str
    lexical_chains_path: str
    embedding_file: str
    hyp_text_path: str
    pair_graph: str
    labels_path: str


class Config:
    def __init__(self, **kwargs):

        self.stage = kwargs.get("stage", "classification")
        self.mode = kwargs.get("mode", "train")
        # 基础路径配置
        self.base_dir = kwargs.get("base_dir", "/mnt/nlp/yuanmengying")
        self.batch_file_size = kwargs.get("batch_file_size", 1)

        # 训练相关配置
        self.epochs = kwargs.get("epochs", 7)
        self.batch_size = kwargs.get("batch_size", 10)
        self.save_dir = kwargs.get("save_dir", "checkpoints")
        self.save_interval = kwargs.get("save_interval", 5)
        self.log_interval = kwargs.get("log_interval", 100)
        self.use_tensorboard = kwargs.get("use_tensorboard", True)
        self.tensorboard_dir = kwargs.get("tensorboard_dir", "runs")
        self.eval_interval = kwargs.get("eval_interval", 1)

        # 优化器相关配置
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)
        self.classifier_lr = kwargs.get("classifier_lr", 1e-4)
        self.generator_lr = kwargs.get("generator_lr", 1e-5)
        self.total_steps = kwargs.get("total_steps", 1000)
        self.optimizer_type = kwargs.get("optimizer_type", "adamw")
        self.scheduler_type = kwargs.get("scheduler_type", "linear_warmup")
        self.device = kwargs.get(
            "device", "cuda:2" if torch.cuda.is_available() else "cpu"
        )
        # 模型配置
        self.model_config = kwargs.get(
            "model_config",
            {
                "in_dim": 1024,
                "hidden_dim": 1024,
                "n_classes": 3,
                "rel_names": [
                    "Cause",
                    "Condition",
                    "Contrast",
                    "Explanation",
                    "Elaboration",
                    "Attribution",
                    "Background",
                    "lexical",
                ],
                "model_name": "/mnt/nlp/yuanmengying/models/t5-small",
                "device": self.device,
            },
        )

        # 初始化数据路径
        self._init_data_paths()

    def _init_data_paths(self):
        """初始化所有数据路径"""
        self.paths = {
            "train": DataPaths(
                rst_path=f"{self.base_dir}/ymy/data/cross_document/train/train1/new_rst_result.jsonl",
                model_output_path=f"{self.base_dir}/nli_data_generate/all_generated_hypothesis.json",
                model_output_emb_path=f"{self.base_dir}/ymy/data/cross_document/train/hyp/hypothesis_embeddings.npz",
                evidence_text_output_path=f"{self.base_dir}/ymy/data/cross_document/train/hyp/evidence_text.json",
                lexical_chains_path=f"{self.base_dir}/ymy/data/cross_document/graph_infos/train",
                embedding_file=f"{self.base_dir}/ymy/data/cross_document/11.1_graph_pairs/train/pre",
                hyp_text_path=f"{self.base_dir}/ymy/data/cross_document/train/hyp/hypothesis_text.json",
                pair_graph=f"{self.base_dir}/ymy/data/cross_document/graph_pairs_for_11.1/train",
                labels_path="/mnt/nlp/yuanmengying/ymy/data/cross_document/train/pre/extractions_train_v3.json",
            ),
            "dev": DataPaths(
                rst_path=f"{self.base_dir}/ymy/data/cross_document/dev/dev1/new_rst_result.jsonl",
                model_output_path=f"{self.base_dir}/nli_data_generate/valid_all_generated_hypothesis.json",
                model_output_emb_path=f"{self.base_dir}/ymy/data/cross_document/dev/hyp/hypothesis_embeddings.npz",
                evidence_text_output_path=f"{self.base_dir}/ymy/data/cross_document/dev/hyp/evidence_text.json",
                lexical_chains_path=f"{self.base_dir}/ymy/data/cross_document/graph_infos/dev",
                embedding_file=f"{self.base_dir}/ymy/data/cross_document/11.1_graph_pairs/dev/pre",
                hyp_text_path=f"{self.base_dir}/ymy/data/cross_document/dev/hyp/hypothesis_text.json",
                pair_graph=f"{self.base_dir}/ymy/data/cross_document/graph_pairs_for_11.1/dev",
                labels_path="/mnt/nlp/yuanmengying/ymy/data/cross_document/dev/pre/extractions_dev_v2.json",
            ),
            "test": DataPaths(
                rst_path=f"{self.base_dir}/ymy/data/cross_document/test/test1/new_rst_result.jsonl",
                model_output_path=f"{self.base_dir}/nli_data_generate/test_all_generated_hypothesis.json",
                model_output_emb_path=f"{self.base_dir}/ymy/data/cross_document/test/hyp/hypothesis_embeddings.npz",
                evidence_text_output_path=f"{self.base_dir}/ymy/data/cross_document/test/hyp/evidence_text.json",
                lexical_chains_path=f"{self.base_dir}/ymy/data/cross_document/graph_infos/test",
                embedding_file=f"{self.base_dir}/ymy/data/cross_document/11.1_graph_pairs/test/pre",
                hyp_text_path=f"{self.base_dir}/ymy/data/cross_document/test/hyp/hypothesis_text.json",
                pair_graph=f"{self.base_dir}/ymy/data/cross_document/graph_pairs_for_11.1/test",
                labels_path="/mnt/nlp/yuanmengying/ymy/data/cross_document/test/pre/extractions_test_v2.json",
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置实例"""
        return cls(**config_dict)

    def to_dict(self):
        d = self.__dict__.copy()
        if "device" in d:
            d["device"] = str(d["device"])
        return d

    def get(self, key, default=None):
        return getattr(self, key, default)


def data_model_loader(device):
    # 创建配置实例
    config = Config(
        save_dir="checkpoints/experiment1",
        tensorboard_dir="runs/experiment1",
        optimizer_type="adamw",
        scheduler_type="linear_warmup",
        device=device,
    )

    # 创建数据集
    logging.info("Processing train data")
    train_dataset = RSTDataset(
        config.paths["train"].rst_path,
        config.paths["train"].model_output_path,
        config.paths["train"].model_output_emb_path,
        config.paths["train"].evidence_text_output_path,
        config.paths["train"].lexical_chains_path,
        config.paths["train"].embedding_file,
        config.paths["train"].hyp_text_path,
        config.paths["train"].labels_path,
        config.batch_file_size,
        save_dir=config.paths["train"].pair_graph,
    )

    logging.info("Processing dev data")
    dev_dataset = RSTDataset(
        config.paths["dev"].rst_path,
        config.paths["dev"].model_output_path,
        config.paths["dev"].model_output_emb_path,
        config.paths["dev"].evidence_text_output_path,
        config.paths["dev"].lexical_chains_path,
        config.paths["dev"].embedding_file,
        config.paths["dev"].hyp_text_path,
        config.paths["dev"].labels_path,
        config.batch_file_size,
        save_dir=config.paths["dev"].pair_graph,
    )

    logging.info("Processing test data")
    test_dataset = RSTDataset(
        config.paths["test"].rst_path,
        config.paths["test"].model_output_path,
        config.paths["test"].model_output_emb_path,
        config.paths["test"].evidence_text_output_path,
        config.paths["test"].lexical_chains_path,
        config.paths["test"].embedding_file,
        config.paths["test"].hyp_text_path,
        config.paths["test"].labels_path,
        config.batch_file_size,
        save_dir=config.paths["test"].pair_graph,
    )
    config.total_steps = (
        train_dataset.file_offsets[0][1] // config.batch_size * config.epochs
    )  # data_loader的长度乘以epochs
    config.warmup_steps = int(config.total_steps * config.warmup_ratio)
    # 初始化模型
    return config, train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    config, train_dataset, dev_dataset, test_dataset = data_model_loader()
