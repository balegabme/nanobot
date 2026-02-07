"""LiteLLM provider implementation for multi-provider support."""

import json
import os
from typing import Any

import httpx
import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


# OpenCode Zen API configuration
OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
OPENCODE_USER_AGENT = "opencode/1.0.0"


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, OpenCode Zen, and many 
    other providers through a unified interface.
    
    Note: OpenCode Zen requires direct HTTP calls (not LiteLLM) to properly
    send the User-Agent header needed to bypass Cloudflare protection.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5"
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        
        # Provider detection flags
        self.is_openrouter = self._detect_openrouter(api_key, api_base)
        self.is_opencode = self._detect_opencode(api_base, default_model)
        self.is_vllm = bool(api_base) and not self.is_openrouter and not self.is_opencode
        
        # Configure environment for LiteLLM
        self._configure_env(api_key, api_base, default_model)
        
        if api_base and not self.is_opencode:
            litellm.api_base = api_base
    
    def _detect_openrouter(self, api_key: str | None, api_base: str | None) -> bool:
        """Detect if using OpenRouter provider."""
        return (
            (api_key and api_key.startswith("sk-or-")) or
            (api_base and "openrouter" in api_base)
        )
    
    def _detect_opencode(self, api_base: str | None, model: str) -> bool:
        """Detect if using OpenCode Zen provider."""
        return (
            (api_base and "opencode.ai" in api_base) or
            model.lower().startswith("opencode/")
        )
    
    def _configure_env(self, api_key: str | None, api_base: str | None, model: str) -> None:
        """Configure environment variables for LiteLLM based on provider."""
        if not api_key:
            return
            
        model_lower = model.lower()
        
        if self.is_openrouter:
            os.environ["OPENROUTER_API_KEY"] = api_key
        elif self.is_vllm or self.is_opencode:
            os.environ["OPENAI_API_KEY"] = api_key
        elif "deepseek" in model_lower:
            os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
        elif "anthropic" in model_lower or "claude" in model_lower:
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif "openai" in model_lower or "gpt" in model_lower:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        elif "gemini" in model_lower:
            os.environ.setdefault("GEMINI_API_KEY", api_key)
        elif any(k in model_lower for k in ("zhipu", "glm", "zai")):
            os.environ.setdefault("ZHIPUAI_API_KEY", api_key)
        elif "groq" in model_lower:
            os.environ.setdefault("GROQ_API_KEY", api_key)
        elif "moonshot" in model_lower or "kimi" in model_lower:
            os.environ.setdefault("MOONSHOT_API_KEY", api_key)
            os.environ.setdefault("MOONSHOT_API_BASE", api_base or "https://api.moonshot.cn/v1")
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Routes to OpenCode Zen direct HTTP or LiteLLM based on provider detection.
        """
        model = model or self.default_model
        
        # OpenCode Zen: Use direct HTTP (required for User-Agent header)
        if self.is_opencode or model.startswith("opencode/"):
            return await self._opencode_chat(
                model=model.replace("opencode/", ""),
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        

        
        # All other providers: Use LiteLLM
        return await self._litellm_chat(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    async def _litellm_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send chat request via LiteLLM."""
        # Apply model prefix based on provider
        model = self._apply_model_prefix(model)
        
        # kimi-k2.5 only supports temperature=1.0
        if "kimi-k2.5" in model.lower():
            temperature = 1.0

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = await acompletion(**kwargs)
            return self._parse_litellm_response(response)
        except Exception as e:
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )
    
    def _apply_model_prefix(self, model: str) -> str:
        """Apply correct LiteLLM prefix based on provider."""
        model_lower = model.lower()
        
        if self.is_openrouter and not model.startswith("openrouter/"):
            return f"openrouter/{model}"
        
        if self.is_vllm:
            return f"hosted_vllm/{model}"
        
        # Standard provider prefixes
        if ("glm" in model_lower or "zhipu" in model_lower) and not model.startswith(("zhipu/", "zai/")):
            return f"zai/{model}"
        
        if ("moonshot" in model_lower or "kimi" in model_lower) and not model.startswith("moonshot/"):
            return f"moonshot/{model}"
        
        if "gemini" in model_lower and not model.startswith("gemini/"):
            return f"gemini/{model}"
        
        return model
    
    def _parse_litellm_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
    
    # =========================================================================
    # OpenCode Zen Direct HTTP Implementation
    # =========================================================================
    
    async def _opencode_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Direct HTTP call to OpenCode Zen API.
        
        OpenCode Zen requires a User-Agent header to bypass Cloudflare protection.
        LiteLLM's extra_headers parameter doesn't reliably pass this header,
        so we use direct HTTP requests instead.
        """
        url = self._get_opencode_url()
        headers = self._get_opencode_headers()
        
        # kimi-k2.5 only supports temperature=1.0
        if "kimi-k2.5" in model.lower():
            temperature = 1.0
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
            
            return self._parse_opencode_response(result)
            
        except httpx.HTTPStatusError as e:
            return LLMResponse(
                content=f"OpenCode API error {e.response.status_code}: {e.response.text}",
                finish_reason="error",
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error calling OpenCode: {str(e)}",
                finish_reason="error",
            )
    
    def _get_opencode_url(self) -> str:
        """Get OpenCode Zen API URL."""
        base = self.api_base or OPENCODE_BASE_URL
        if not base.endswith("/chat/completions"):
            base = f"{base.rstrip('/')}/chat/completions"
        return base
    
    def _get_opencode_headers(self) -> dict[str, str]:
        """Get headers for OpenCode Zen API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": OPENCODE_USER_AGENT,
        }
    
    def _parse_opencode_response(self, result: dict[str, Any]) -> LLMResponse:
        """Parse OpenCode Zen API response into standard format."""
        choice = result["choices"][0]
        message = choice["message"]
        
        # Parse tool calls
        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCallRequest(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args,
                ))
        
        # Get content (handle reasoning models that return reasoning instead of content)
        content = message.get("content") or ""
        if not content and message.get("reasoning"):
            content = message["reasoning"]
        
        # Parse usage (optional, some responses may not include it)
        usage = {}
        if result.get("usage"):
            usage_data = result["usage"] or {}
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=usage,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model


