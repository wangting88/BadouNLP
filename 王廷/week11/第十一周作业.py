# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

"""
基于BERT的Seq2Seq生成模型（监督式微调）
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path, sep_token_id):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding部分的损失
        self.sep_token_id = sep_token_id  # 保存[SEP]标记的ID

    def forward(self, x, y=None):
        # 训练模式：计算损失
        if y is not None:
            # 构建下三角注意力掩码（防止看到未来信息）
            seq_len = x.shape[1]
            mask = torch.tril(torch.ones((x.shape[0], seq_len, seq_len)))
            if torch.cuda.is_available():
                mask = mask.cuda()

            outputs, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(outputs)  # [batch_size, seq_len, vocab_size]

            # 只计算正文部分的损失（跳过[CLS]和标题）
            # 假设输入格式: [CLS] + 标题 + [SEP] + 正文
            sep_idx = (x == self.sep_token_id).nonzero(as_tuple=True)
            loss_mask = torch.zeros_like(y)

            # 确保每个样本都有找到[SEP]标记
            if len(sep_idx[0]) > 0:
                # 获取每个样本中第一个[SEP]的位置
                batch_indices = sep_idx[0]
                token_indices = sep_idx[1]

                # 为每个样本创建位置映射
                for i in range(x.size(0)):
                    # 找到当前样本的所有[SEP]位置
                    sample_sep = token_indices[batch_indices == i]
                    if len(sample_sep) > 0:
                        first_sep = sample_sep[0]  # 取第一个[SEP]
                        if first_sep + 1 < y.size(1):  # 确保不越界
                            loss_mask[i, first_sep + 1:] = 1  # 从[SEP]后开始计算损失

            # 计算掩码损失
            active_loss = loss_mask.view(-1) == 1
            active_logits = y_pred.view(-1, y_pred.shape[-1])[active_loss]
            active_labels = y.view(-1)[active_loss]

            if active_logits.numel() > 0:  # 确保有激活的损失
                return self.loss(active_logits, active_labels)
            else:
                return torch.tensor(0.0, device=x.device)  # 返回零损失

        # 生成模式：返回预测概率
        else:
            # 预测时同样使用掩码
            seq_len = x.shape[1]
            mask = torch.tril(torch.ones((x.shape[0], seq_len, seq_len)))
            if torch.cuda.is_available():
                mask = mask.cuda()
            outputs, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(outputs)
            return torch.softmax(y_pred, dim=-1)


# 加载JSON语料
def load_json_corpus(json_path):
    samples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            samples.append((data["title"], data["content"]))
    return samples


# 构建单条训练样本
def build_sample(tokenizer, title, content, max_length=128):
    # 输入格式: [CLS] + 标题 + [SEP] + 正文
    input_text = f"[CLS]{title}[SEP]{content}"
    input_ids = tokenizer.encode(
        input_text,
        add_special_tokens=False,  # 已手动添加特殊标记
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )

    # 目标输出：正文部分（忽略标题）
    target_ids = tokenizer.encode(
        content,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    return torch.LongTensor(input_ids), torch.LongTensor(target_ids)


# 构建数据集
class JsonDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        title, content = self.samples[idx]
        return build_sample(self.tokenizer, title, content, self.max_length)


# 文本生成函数
def generate_text(model, tokenizer, title, max_length=50, temperature=1.0):
    model.eval()
    # 初始输入: [CLS] + 标题 + [SEP]
    input_text = f"[CLS]{title}[SEP]"
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    generated = []

    for _ in range(max_length):
        x = torch.LongTensor([input_ids])
        if torch.cuda.is_available():
            x = x.cuda()

        with torch.no_grad():
            probs = model(x)[0][-1]  # 最后一个位置的预测概率

        probs = probs / temperature
        next_token = torch.multinomial(torch.softmax(probs, dim=-1), 1).item()

        # 遇到[SEP]或[PAD]停止
        if next_token == tokenizer.sep_token_id or next_token == tokenizer.pad_token_id:
            break

        generated.append(next_token)
        input_ids.append(next_token)  # 将生成词加入输入

    return tokenizer.decode(generated)


# 训练函数
def train(json_path, save_weight=True):
    epoch_num = 20
    batch_size = 32
    max_length = 512  # 最大序列长度
    pretrain_model_path = r"D:\练习\AI学习\新建文件夹\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # 特殊标记处理（确保[CLS]和[SEP]在词表中）
    if tokenizer.cls_token_id is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.sep_token_id is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    # 加载数据
    samples = load_json_corpus(json_path)
    dataset = JsonDataset(samples, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型 - 传递sep_token_id
    model = LanguageModel(768, tokenizer.vocab_size, pretrain_model_path, tokenizer.sep_token_id)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epoch_num}, Loss: {avg_loss:.4f}")

        # 生成示例
        test_title = "阿根廷歹徒抢服装尺码不对拿回店里换"
        generated_content = generate_text(model, tokenizer, test_title)
        print(f"标题: {test_title}\n生成正文: {generated_content}\n")

    # 保存模型
    if save_weight:
        torch.save(model.state_dict(), "bert_generator.pth")


if __name__ == "__main__":
    train(r"D:\练习\AI学习\新建文件夹\sample_data.json", save_weight=True)
