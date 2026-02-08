"""Test script to isolate OpenCode Gemini 3 issue."""
import asyncio
import httpx

OPENCODE_URL = "https://opencode.ai/zen/v1/chat/completions"
API_KEY = "sk-F9gLGD81sqb74q5VFrjBU3p5UQY8AkSrwNgQgxMkLc8pNrnAodvt66vwepI6IKaY"

async def test_minimal():
    """Test minimal request without tools."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "opencode/1.0.0",
    }
    
    # Test 1: Minimal request
    print("=== Test 1: Minimal request (no tools) ===")
    payload = {
        "model": "gemini-3-flash",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 100,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPENCODE_URL, headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    
    # Test 2: With tools
    print("\n=== Test 2: With tools ===")
    payload["tools"] = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPENCODE_URL, headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    
    # Test 3: Compare with kimi
    print("\n=== Test 3: kimi-k2.5 with same request ===")
    payload["model"] = "kimi-k2.5"
    payload["temperature"] = 1.0  # kimi requires this
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPENCODE_URL, headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")

if __name__ == "__main__":
    asyncio.run(test_minimal())
