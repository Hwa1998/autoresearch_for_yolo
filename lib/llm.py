"""
统一 LLM 客户端 - 支持 DeepSeek / OpenAI / Anthropic Claude。
"""
import json
import requests
from dataclasses import dataclass


@dataclass
class LLMConfig:
    endpoint: str
    api_key: str
    model: str
    timeout: int = 60


class LLMClient:
    """多后端 LLM 客户端，自动适配 API 格式。"""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        # 规范化 model 名称：去除 owner/ 前缀
        if "/" in cfg.model:
            cfg.model = cfg.model.split("/")[-1]

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str | None:
        """发送对话请求，返回 LLM 响应文本。"""
        cfg = self.cfg
        is_anthropic = "anthropic" in (cfg.endpoint or "") or "claude" in (cfg.model or "").lower()

        if is_anthropic:
            return self._call_anthropic(system_prompt + "\n\n" + user_prompt)

        return self._call_openai_compatible(system_prompt, user_prompt, temperature, max_tokens)

    def _call_openai_compatible(self, system: str, user: str, temp: float, max_tok: int) -> str | None:
        url = self._build_url()
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temp,
            "max_tokens": max_tok,
        }
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=self.cfg.timeout)
            if not r.ok:
                print(f"LLM API error {r.status_code}: {r.text[:200]}")
                return None
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM request error: {e}")
            return None

    def _call_anthropic(self, prompt: str) -> str | None:
        headers = {"x-api-key": self.cfg.api_key, "Content-Type": "application/json"}
        body = {
            "model": self.cfg.model,
            "prompt": prompt,
            "max_tokens_to_sample": 2000,
            "temperature": 0.3,
        }
        try:
            r = requests.post(self.cfg.endpoint, headers=headers, data=json.dumps(body), timeout=self.cfg.timeout)
            if not r.ok:
                print(f"Anthropic API error {r.status_code}: {r.text[:200]}")
                return None
            j = r.json()
            return j.get("completion") or j.get("completion_text") or r.text
        except Exception as e:
            print(f"Anthropic request error: {e}")
            return None

    def _build_url(self) -> str:
        url = self.cfg.endpoint
        if not url:
            return url
        # DeepSeek base URL → 自动追加 /v1/chat/completions
        if url.rstrip("/").endswith("api.deepseek.com"):
            return url.rstrip("/") + "/v1/chat/completions"
        return url
