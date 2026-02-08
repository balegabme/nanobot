"""OpenAI SDK provider implementation for OpenCode API."""

import json
from typing import Any

from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class OpenAISDKProvider(LLMProvider):
    """
    LLM provider using the official OpenAI SDK.
    
    This provider uses the OpenAI Python SDK which handles all the
    request formatting, streaming, and error handling properly.
    Works with OpenCode Zen API and any OpenAI-compatible endpoint.
    
    All configuration (api_key, api_base, model) must come from config.json.
    """
    
    def __init__(
        self, 
        api_key: str, 
        api_base: str,
        default_model: str,
    ):
        """
        Initialize the OpenAI SDK provider.
        
        Args:
            api_key: API key from config (required)
            api_base: API base URL from config (required)
            default_model: Default model from config (required)
        
        Raises:
            ValueError: If api_key, api_base, or default_model is empty/missing
        """
        if not api_key:
            raise ValueError("api_key is required. Configure it in config.json under providers.<provider>.apiKey")
        if not api_base:
            raise ValueError("api_base is required. Configure it in config.json under providers.<provider>.apiBase")
        if not default_model:
            raise ValueError("default_model is required. Configure it in config.json under agents.defaults.model")
        
        super().__init__(api_key, api_base)
        self.default_model = default_model
        
        # Initialize the AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={
                "User-Agent": "opencode/1.0.0"  # Required for Cloudflare bypass
            }
        )
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request using the OpenAI SDK.
        """
        model = model or self.default_model
        
        # Strip provider prefix if present (e.g., opencode/gemini-3-flash -> gemini-3-flash)
        if "/" in model:
            model = model.split("/", 1)[1]
        
        # kimi-k2.5 only supports temperature=1.0
        if "kimi-k2.5" in model.lower():
            temperature = 1.0
        
        try:
            # Build kwargs
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            # Make the API call using the SDK
            response = await self.client.chat.completions.create(**kwargs)
            
            return self._parse_response(response)
            
        except Exception as e:
            return LLMResponse(
                content=f"Error calling OpenAI SDK: {str(e)}",
                finish_reason="error",
            )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI SDK response into standard format."""
        choice = response.choices[0]
        message = choice.message
        
        # Parse tool calls
        tool_calls = []
        if message.tool_calls:
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
        
        # Get content
        content = message.content or ""
        
        # Parse usage (handle missing usage gracefully)
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
