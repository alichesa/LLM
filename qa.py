import requests

url = "http://127.0.0.1:5005/chat"
data = {"input": "宝宝晚上一直哭，而且我想知道现在几点了"}
response = requests.post(url, json=data)

print(response.json()["response"])
