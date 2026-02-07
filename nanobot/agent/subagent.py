"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.browser import BrowserTool


class SubagentManager:
    """
    Manages background subagent execution with configurable model, prompt, and tools.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.default_model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
    
    async def spawn(
        self,
        task: str,
        label: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a subagent with optional configuration.
        
        Args:
            task: The task description.
            label: Display label.
            model: LLM model override.
            system_prompt: Custom system prompt.
            tools: List of tool names to enable.
            origin_channel: Channel to announce results.
            origin_chat_id: Chat ID to announce results.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        
        config = {
            "model": model or self.default_model,
            "system_prompt": system_prompt,
            "tools": tools,
            "origin": {"channel": origin_channel, "chat_id": origin_chat_id},
        }
        
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, config)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))
        
        logger.info(f"Spawned subagent [{task_id}]: {display_label} (model={config['model']})")
        return f"Subagent [{display_label}] started (id: {task_id}). Will notify on completion."
    
    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        config: dict[str, Any],
    ) -> None:
        """Execute the subagent task."""
        logger.info(f"Subagent [{task_id}] starting: {label}")
        
        try:
            tools = self._build_tools(config.get("tools"))
            prompt = config.get("system_prompt") or self._build_default_prompt(task)
            model = config["model"]
            
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": task},
            ]
            
            max_iterations = 15
            final_result: str | None = None
            
            for _ in range(max_iterations):
                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=model,
                )
                
                if response.has_tool_calls:
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })
                    
                    for tc in response.tool_calls:
                        logger.debug(f"Subagent [{task_id}] exec: {tc.name}")
                        result = await tools.execute(tc.name, tc.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break
            
            if final_result is None:
                final_result = "Task completed but no final response generated."
            
            logger.info(f"Subagent [{task_id}] completed")
            await self._announce_result(task_id, label, task, final_result, config["origin"], "ok")
            
        except Exception as e:
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, f"Error: {e}", config["origin"], "error")
    
    def _build_tools(self, tool_names: list[str] | None) -> ToolRegistry:
        """Build tool registry based on allowed tool names."""
        registry = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        
        # All available tools
        all_tools = {
            "read_file": ReadFileTool(allowed_dir=allowed_dir),
            "write_file": WriteFileTool(allowed_dir=allowed_dir),
            "list_dir": ListDirTool(allowed_dir=allowed_dir),
            "exec_shell": ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ),
            "web_search": WebSearchTool(api_key=self.brave_api_key),
            "web_fetch": WebFetchTool(),
            "browser": BrowserTool(workspace=self.workspace),
        }
        
        # Filter if specific tools requested
        names_to_use = tool_names if tool_names else list(all_tools.keys())
        
        for name in names_to_use:
            if name in all_tools:
                registry.register(all_tools[name])
        
        return registry
    
    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce result via message bus."""
        status_text = "completed" if status == "ok" else "failed"
        
        content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize naturally for the user. Keep brief (1-2 sentences)."""
        
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=content,
        )
        
        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced to {origin['channel']}:{origin['chat_id']}")
    
    def _build_default_prompt(self, task: str) -> str:
        """Build default subagent system prompt."""
        return f"""# Subagent

You are a subagent spawned to complete a specific task.

## Your Task
{task}

## Rules
1. Stay focused - complete only the assigned task
2. Your final response will be reported to the main agent
3. Be concise but informative

## Workspace
{self.workspace}

When done, provide a clear summary of findings or actions."""
    
    def get_running_count(self) -> int:
        """Return number of running subagents."""
        return len(self._running_tasks)
