# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",
    "max_length": 128,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 30,
    "batch_size": 16,
    "optimizer": "adam",
    # "learning_rate": 1e-3,
    "learning_rate": 3e-5,
    "use_crf": False,
    "class_num": 9,
    "pooling_style": "max",
    "pretrain_model_path": r"D:\BaiduNetdiskDownload\AI\nlp\bert-base-chinese"
}
