"""Spawn tool for creating background subagents."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


# Available tools that subagents can use
AVAILABLE_SUBAGENT_TOOLS = [
    "read_file", "write_file", "list_dir",
    "exec_shell", "web_search", "web_fetch", "browser"
]


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""
    
    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
    
    @property
    def name(self) -> str:
        return "spawn"
    
    @property
    def description(self) -> str:
        return (
            "Spawn a subagent for background task execution. "
            "Configure model, system_prompt, and tools for cost efficiency. "
            f"Available tools: {', '.join(AVAILABLE_SUBAGENT_TOOLS)}"
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Short label for display",
                },
                "model": {
                    "type": "string",
                    "description": "LLM model to use (e.g. 'gpt-4o-mini' for cost savings)",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Custom system prompt for the subagent",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Tools to enable: {', '.join(AVAILABLE_SUBAGENT_TOOLS)}",
                },
            },
            "required": ["task"],
        }
    
    async def execute(
        self,
        task: str,
        label: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        """Spawn a subagent with optional configuration."""
        # Validate tools if provided
        if tools:
            invalid = [t for t in tools if t not in AVAILABLE_SUBAGENT_TOOLS]
            if invalid:
                return f"Error: Invalid tools: {invalid}. Available: {AVAILABLE_SUBAGENT_TOOLS}"
        
        return await self._manager.spawn(
            task=task,
            label=label,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
