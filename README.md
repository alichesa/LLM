# LLM

```
代码在我的Github里面，由于文件大小没有上传大模型的权重，需要自行在Hugging Face进行下载
GIthub：https://github.com/alichesa/LLM/edit/main/README.md
LLM：(https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat/tree/main)
```

# 模型的选择
 - 尝试了chatgpt2、llama模型等，模型层次不齐，同时为了是使得自己的显卡能够运行模型，最后选择在中文领域口碑比较好的Qwen1.5-1.8B-Chat进行实例化


# 功能
- 利用轻量级Web应用框架Flask构建HTTP服务器聊天路由，并利用Post进行传递
  ![[Pasted image 20250614172038.png]]

- 用一个简单的prompt进行约束，使其初始化自己的身份
```
system_prompt = {"role": "system", "content": "你是一个专业的中文育儿顾问，请准确、温柔地回答家长的问题。"}
```

- 定义一个简单的历史记录系统，将QA都转换为json存储
  ![[Pasted image 20250614172203.png]]

- 利用大模型生成RAG库，将LLM链接好RAG进行综合输入
```
with open("RAG.txt", "r", encoding="utf-8") as f:  
    corpus = [line.strip() for line in f if line.strip()]
```

# 有待完善的地方
- 利用爬虫或者大预言模型生成育儿领域的QA，然后利用LoRA进行低秩微调，从而生成领域对应专业的助手
- Agent逻辑不够清晰，没有体现出智能体的最根本的“Re-act”
- prompt提示词工程过于简单
