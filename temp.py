"""Minimal DashScope-compatible chat smoke test.

Configure LLM_API_KEY and optionally LLM_BASE_URL before running this file.
"""
import os

from openai import OpenAI


client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

completion = client.chat.completions.create(
    model=os.getenv("LLM_MODEL", "qwen-plus"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请回复：连接成功。"},
    ],
)
print(completion.choices[0].message.content)