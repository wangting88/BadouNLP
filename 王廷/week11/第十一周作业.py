import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertLMHeadModel, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random


# 设置随机种子确保可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# 1. 数据加载与预处理
class NewsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        title = item['title']
                        content = item['content']
                        self.data.append((title, content))
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title, content = self.data[idx]

        # 构建输入格式：标题 + 分隔符 + 正文
        input_text = f"标题：{title} 正文：{content}"
        # 编码
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # 提取输入ID和注意力掩码
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # 构建标签（仅正文部分需要计算损失）
        title_part = f"标题：{title} 正文："
        title_len = len(self.tokenizer.encode(title_part, add_special_tokens=False))
        labels = input_ids.clone()
        labels[:title_len] = -100  # 标题部分不计算损失

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# 2. 模型定义
def load_model_and_tokenizer(model_name=r"D:\练习\AI学习\新建文件夹\bert-base-chinese"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # 加载模型时指定is_decoder=True，解决警告问题
    model = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)
    return model, tokenizer


# 3. 训练函数
def train(model, train_loader, optimizer, scheduler, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # BertLMHeadModel返回的是元组，第一个元素是loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 从元组中获取loss（第一个元素）
            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")


# 4. 生成函数
def generate_content(title, model, tokenizer, device, max_length=200):
    model.eval()
    input_text = f"标题：{title} 正文："
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,  # 启用采样模式（与temperature配合）
            temperature=0.7,
            top_k=50,
            num_beams=1,    # 明确单 beam 模式
            early_stopping=False  # 单 beam 时关闭 early_stopping
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    content = generated_text.split("正文：")[-1].strip()
    return content


# 5. 主函数
def main():
    # 配置参数
    file_path = r"D:\练习\AI学习\新建文件夹\sample_data.json"
    model_name = r"D:\练习\AI学习\新建文件夹\bert-base-chinese"
    batch_size = 2
    learning_rate = 2e-5
    epochs = 20
    max_length = 512

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to(device)

    # 加载数据集
    dataset = NewsDataset(file_path, tokenizer, max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器 - 使用PyTorch原生的AdamW避免警告
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练模型
    print("开始训练...")
    train(model, train_loader, optimizer, scheduler, device, epochs)

    # 保存模型
    torch.save(model.state_dict(), "title2content_model.pt")
    print("模型保存完成")

    # 测试生成
    test_title = "人工智能助力环境保护"
    generated_content = generate_content(test_title, model, tokenizer, device)
    print(f"\n测试生成:")
    print(f"标题: {test_title}")
    print(f"生成正文: {generated_content}")


if __name__ == "__main__":
    main()
