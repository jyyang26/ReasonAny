#!/usr/bin/env python3
import openai
import sys

# --- 配置 ---
API_KEY = "sk-"
BASE_URL = ""
MODEL_TO_TEST = "o4-mini"
PROMPT_TO_TEST = "hello"

# --- 1. 初始化客户端 ---
try:
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    print(f"Attempting to connect to: {BASE_URL}...")
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. 发送测试请求 ---
try:
    print(f"Sending prompt '{PROMPT_TO_TEST}' to model '{MODEL_TO_TEST}'...")
    
    response = client.chat.completions.create(
        model=MODEL_TO_TEST,
        messages=[{"role": "user", "content": PROMPT_TO_TEST}],
        temperature=0.0,
        max_tokens=1024 # 限制返回长度
    )
    
    # --- 3. 打印成功响应 ---
    print("\n--- SUCCESS: API call successful! ---")
    
    # 打印完整的 response 对象，以便调试
    print("\n[Full Response Object]:")
    print(response)
    
    # 提取并打印消息内容
    if response.choices:
        message_content = response.choices[0].message.content
        print("\n[Model's Reply]:")
        print(message_content)
    else:
        print("\n[Warning] API returned a response, but no 'choices' were found.")

except openai.AuthenticationError as e:
    print("\n--- FAILED: Authentication Error ---", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    print("Please check if your API_KEY is correct or has expired.", file=sys.stderr)

except openai.APINotFoundError as e:
    print("\n--- FAILED: API Not Found Error ---", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    print("This often means the BASE_URL is incorrect or the server is not running.", file=sys.stderr)

except openai.RateLimitError as e:
    print("\n--- FAILED: Rate Limit Error ---", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    print("You have exceeded your API quota.", file=sys.stderr)

except openai.APIConnectionError as e:
    print("\n--- FAILED: Connection Error ---", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    print("Could not connect to the API. Check your network or the server status.", file=sys.stderr)

except Exception as e:
    print(f"\n--- FAILED: An unexpected error occurred ---", file=sys.stderr)
    print(f"Error Type: {type(e).__name__}", file=sys.stderr)
    print(f"Error Details: {e}", file=sys.stderr)