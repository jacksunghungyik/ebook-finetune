#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量TXT电子书微调脚本
基于 Unsloth 和 Qwen 模型
"""

import os
import torch
from typing import List, Dict
import json

class Config:
    """配置参数"""
    MODEL_NAME = "unsloth/qwen2.5-14b-bnb-4bit"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    
    # LoRA配置
    LORA_RANK = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    
    # 训练配置
    OUTPUT_DIR = "./qwen_ebook_finetuned"
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # 数据配置
    CHUNK_SIZE = 1024
    MIN_CHUNK_LENGTH = 50
    BOOKS_FOLDER = "./books"
    MAX_BOOKS = None
    MERGE_BOOKS = True
    SUPPORTED_FORMATS = [".txt", ".md"]

def main():
    """主函数"""
    print("📚 批量TXT电子书微调工具")
    print("请先安装依赖：pip install -r requirements.txt")
    
    config = Config()
    
    # 创建books文件夹
    if not os.path.exists(config.BOOKS_FOLDER):
        os.makedirs(config.BOOKS_FOLDER)
        print(f"📁 已创建文件夹: {config.BOOKS_FOLDER}")
        
        # 创建使用说明
        readme_content = """📚 电子书文件夹使用说明

请将您的txt电子书文件放在此文件夹中。

支持格式：
- .txt 文件（推荐UTF-8编码）
- .md 文件

使用方法：
1. 将电子书文件放入此文件夹
2. 运行：python batch_finetune.py
3. 等待处理完成

祝您微调愉快！🚀"""
        
        with open(os.path.join(config.BOOKS_FOLDER, "README.txt"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("请将txt电子书文件放入books文件夹，然后重新运行脚本")
        return
    
    print("✅ 配置完成，可以开始微调！")
    print("注意：需要先安装所有依赖包")

if __name__ == "__main__":
    main()
