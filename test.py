import os
import json
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# === 加载知识库文本 ===
with open("RAG.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(corpus)

def retrieve_context(query, top_k=3):
    """从知识库中检索最相关的段落"""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return "\n".join([corpus[i] for i in top_indices])

# === 定义 Agent 工具 ===
def tool_rag(query):
    retrieved = retrieve_context(query)
    return f"【知识库资料】\n{retrieved}\n"

def tool_time(_):
    return f"【时间助手】现在的时间是 {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}。\n"

def tool_calc(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"【计算器结果】你要的结果是：{result}\n"
    except:
        return "【计算器错误】无法识别或不安全的表达式。\n"


# === Agent 决策器 ===
def decide_and_use_tool(user_input):
    if "现在几点" in user_input or "时间" in user_input:
        return tool_time(user_input)
    elif any(k in user_input for k in ["宝宝", "辅食", "喂养", "啼哭", "睡眠", "育儿"]):
        return tool_rag(user_input)
    elif any(k in user_input for k in ["加", "减", "乘", "除", "+", "-", "*", "/"]):
        return tool_calc(user_input)
    else:
        return ""


# === Flask 应用启动 ===
app = Flask(__name__)

# 加载模型
model_path = "Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda").eval()

# 历史记录缓存
history_dir = "history"
os.makedirs(history_dir, exist_ok=True)
memory_store = {}

def load_history(user_id):
    path = os.path.join(history_dir, f"{user_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            memory_store[user_id] = json.load(f)
    else:
        memory_store[user_id] = []

def save_history(user_id):
    path = os.path.join(history_dir, f"{user_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory_store[user_id], f, ensure_ascii=False, indent=2)


# === 聊天接口 ===
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("input", "").strip()
        user_id = data.get("user_id", "default_user")
        clear_history = data.get("clear", False)

        if not user_input:
            return jsonify({"error": "输入不能为空"}), 400

        if clear_history:
            memory_store[user_id] = []
        elif user_id not in memory_store:
            load_history(user_id)

        # System prompt：角色设定
        system_prompt = {
            "role": "system",
            "content": "你是一个专业的中文育儿顾问，请准确、温柔地回答家长的问题。"
        }

        # Agent 工具调用
        tool_output = decide_and_use_tool(user_input)

        # 构造 messages
        history = memory_store[user_id][-10:]  # 限制历史长度
        user_message = {"role": "user", "content": (tool_output + user_input)}
        messages = [system_prompt] + history + [user_message]

        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # 存储历史
        memory_store[user_id].append({"role": "user", "content": user_input})
        memory_store[user_id].append({"role": "assistant", "content": response})
        save_history(user_id)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
