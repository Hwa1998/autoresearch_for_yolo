import requests
import json

api_key = "sk-xxx"
endpoint = "https://api.deepseek.com"
url = endpoint.rstrip('/') + '/v1/chat/completions'
print('POST', url)

data = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello from test call"}
    ],
    "stream": False
}
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
try:
    r = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    print('Status:', r.status_code)
    print('Response:', r.text[:4000])
except Exception as e:
    print('Request failed:', e)
