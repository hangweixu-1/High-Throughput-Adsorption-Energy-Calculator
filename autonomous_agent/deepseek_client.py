import json, urllib.request, urllib.error

class DeepSeekClient:
    def __init__(self, api_key, model="deepseek-chat", base_url="https://api.deepseek.com"):
        self.api_key=api_key
        self.model=model
        self.base_url=base_url.rstrip("/")
        self.calls=0
        self.total_tokens=0

    def chat(self, messages, temperature=0.2, max_tokens=1200):
        self.calls += 1
        url=self.base_url + "/chat/completions"
        payload={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        data=json.dumps(payload).encode("utf-8")
        req=urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type","application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                j=json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body=e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek HTTPError {e.code}: {body}")
        usage=j.get("usage") or {}
        if "total_tokens" in usage:
            self.total_tokens += int(usage.get("total_tokens",0))
        choices=j.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""
