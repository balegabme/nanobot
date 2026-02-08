"""LiteLLM provider implementation for multi-provider support."""

import json
import os
from typing import Any

import httpx
import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, OpenCode Zen, and many 
    other providers through a unified interface.
    
    Provider-specific logic is driven by the registry (see providers/registry.py).
    OpenCode Zen requires direct HTTP calls to properly send the User-Agent header
    needed to bypass Cloudflare protection.
    """
    
    def __init__(
        self, 
        api_key: str, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
    ):
        """
        Initialize the LiteLLM provider.
        
        Args:
            api_key: API key from config (required)
            api_base: API base URL from config (optional for most providers)
            default_model: Default model from config (required)
            extra_headers: Custom headers (e.g. APP-Code for AiHubMix)
        
        Raises:
            ValueError: If api_key or default_model is empty/missing
        """
        if not api_key:
            raise ValueError("api_key is required. Configure it in config.json under providers.<provider>.apiKey")
        if not default_model:
            raise ValueError("default_model is required. Configure it in config.json under agents.defaults.model")
        
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        
        # Detect gateway / local deployment from api_key and api_base
        self._gateway = find_gateway(api_key, api_base)
        
        # Backwards-compatible flags (used by tests and possibly external code)
        self.is_openrouter = bool(self._gateway and self._gateway.name == "openrouter")
        self.is_aihubmix = bool(self._gateway and self._gateway.name == "aihubmix")
        self.is_opencode = bool(self._gateway and self._gateway.name == "opencode")
        self.is_vllm = bool(self._gateway and self._gateway.is_local)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        if api_base and not self.is_opencode:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        if self._gateway:
            # Gateway / local: direct set (not setdefault)
            os.environ[self._gateway.env_key] = api_key
            return
        
        # Standard provider: match by model name
        spec = find_by_model(model)
        if spec:
            os.environ.setdefault(spec.env_key, api_key)
            # Resolve env_extras placeholders:
            #   {api_key}  → user's API key
            #   {api_base} → user's api_base, falling back to spec.default_api_base
            effective_base = api_base or spec.default_api_base
            for env_name, env_val in spec.env_extras:
                resolved = env_val.replace("{api_key}", api_key)
                resolved = resolved.replace("{api_base}", effective_base)
                os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        # OpenCode uses direct HTTP, don't transform model name for LiteLLM
        if self.is_opencode:
            # Strip opencode/ prefix if present
            if model.lower().startswith("opencode/"):
                model = model.split("/", 1)[1]
            return model
        
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        
        return model
    
    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
        # Also check gateway overrides (e.g., opencode kimi-k2.5)
        if self._gateway:
            for pattern, overrides in self._gateway.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
    
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
        raw_model = model or self.default_model
        
        # OpenCode Zen: Use direct HTTP (required for User-Agent header)
        if self.is_opencode or raw_model.lower().startswith("opencode/"):
            # Strip prefix for OpenCode
            clean_model = raw_model
            if clean_model.lower().startswith("opencode/"):
                clean_model = clean_model.split("/", 1)[1]
            return await self._opencode_chat(
                model=clean_model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        # All other providers: Use LiteLLM
        model = self._resolve_model(raw_model)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_base directly for custom endpoints (vLLM, etc.)
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
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
        """Get OpenCode Zen API URL from configured api_base."""
        if not self.api_base:
            # Use default from registry
            from nanobot.providers.registry import find_by_name
            spec = find_by_name("opencode")
            base = spec.default_api_base if spec else "https://opencode.ai/zen/v1"
        else:
            base = self.api_base
        if not base.endswith("/chat/completions"):
            base = f"{base.rstrip('/')}/chat/completions"
        return base
    
    def _get_opencode_headers(self) -> dict[str, str]:
        """Get headers for OpenCode Zen API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "opencode/1.0.0",  # Required for Cloudflare bypass
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
