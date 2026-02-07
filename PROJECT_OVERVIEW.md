# AI Agent Onboarding & Architecture Guide for `nanobot`

> **Designed for AI Agents & Developers**  
> This document provides a deep conceptual and technical map of the `nanobot` project. Use this to orient yourself immediately before modifying code.

---

## 1. Project Identity & Intent

**Nanobot** is an ultra-lightweight, extensible personal AI assistant framework written in Python.

- **Core Philosophy:** Minimalism (~3.5k lines of code), modularity, and "local-first" capability.
- **Primary Goal:** Enable users to run a powerful, multi-modal AI agent that can control their local computer, manage schedules, and interact via various chat platforms (Telegram, WhatsApp, Discord, Feishu).
- **Target Audience:** Developers and researchers who want a clean, hackable agent foundation without the bloat of enterprise frameworks.

---

## 2. System Architecture

The system follows a **Event-Driven, Bus-Centric Architecture**. It decouples the "Brain" (Agent) from the "Mouth/Ears" (Channels).

### High-Level Data Flow

1.  **Input:** User sends a message via a Channel (e.g., Telegram).
2.  **Ingest:** Channel converts this to an `InboundMessage` and pushes it to the `MessageBus`.
3.  **Process:** The `AgentLoop` (running continuously) consumes the `InboundMessage`.
    - It retrieves context (History, Memories).
    - It invokes the LLM (via `Provider`).
    - It executes Tools (Shell, File, Web, etc.).
4.  **Output:** The Agent generates a response, wraps it in an `OutboundMessage`, and pushes it to the `MessageBus`.
5.  **Dispatch:** The `ChannelManager`'s dispatcher picks up the message and routes it back to the specific Channel to be sent to the user.

### Key Components

| Component       | Path                      | Role                                                                               | Key Class                       |
| :-------------- | :------------------------ | :--------------------------------------------------------------------------------- | :------------------------------ |
| **Agent Core**  | `nanobot/agent/loop.py`   | The main event loop. Orchestrates LLM calls, tool execution, and state management. | `AgentLoop`                     |
| **Message Bus** | `nanobot/bus/queue.py`    | Async queue system separating I/O from logic.                                      | `MessageBus`                    |
| **Channels**    | `nanobot/channels/`       | Adapters for external chat platforms.                                              | `ChannelManager`, `BaseChannel` |
| **Skills**      | `nanobot/agent/skills.py` | Implementation of capability loading from Markdown definitions.                    | `SkillsLoader`                  |
| **Providers**   | `nanobot/providers/`      | Abstraction layer for LLM APIs (OpenRouter, OpenAI, vLLM).                         | `LLMProvider`                   |
| **Tools**       | `nanobot/agent/tools/`    | Native Python functions exposed to the LLM (File, Shell, Web, Browser, Secrets).   | `ToolRegistry`                  |

---

## 3. Deep Dive: The Agent Loop (`agent/loop.py`)

This is the heart of the application.

- **Initialization:** Sets up `ContextBuilder`, `SessionManager`, `ToolRegistry`, and `SubagentManager`.
- **The Loop (`run` method):**
  - Waits for `bus.consume_inbound()`.
  - **Context Building:** Constructs the prompt including conversation history and loaded skills.
  - **LLM Interaction:** Sends messages to the Provider.
  - **Tool Execution:** If the LLM requests tool calls, the loop executes them and feeds the result back (ReAct pattern).
  - **Response:** Once the LLM provides a final text response, it's sent to the `bus`.

---

## 4. Skills System (`agent/skills.py`)

Nanobot uses a unique "Markdown-as-Code" approach for skills.

- **Definition:** Skills are defined in `SKILL.md` files living in `skills/` or `~/.nanobot/workspace/skills/`.
- **Metadata:** Frontmatter (YAML) defines requirements (CLI tools, Env vars).
- **Loading:** The `SkillsLoader` parses these files and injects the instructions into the Agent's system prompt or context.
- **Optimization:** The agent can "read" skill definitions on demand rather than having them all in context 24/7.

---

## 5. Directory Structure Guide

```text
nanobot/
├── agent/           # 🧠 CORE LOGIC
│   ├── loop.py      #    Main event loop
│   ├── commands.py  #    Slash command handling
│   ├── context.py   #    Prompt assembly
│   ├── tools/       #    Native tools (Shell, File, Web)
│   └── subagent.py  #    Spawning child agents
├── bus/             # 🚌 MESSAGE PASSING
│   ├── queue.py     #    AsyncIO queues
│   └── events.py    #    Data classes (InboundMessage)
├── channels/        # 🔌 CONNECTORS
│   ├── manager.py   #    Lifecycle management of channels
│   └── [platform]/  #    Specific implementations (telegram.py, etc.)
├── skills/          # 📚 CAPABILITIES
│   └── [skill_name]/SKILL.md  # Instruction sets
├── providers/       # 🤖 MODEL ADAPTERS
├── config/          # ⚙️ SETTINGS
└── cli/             # 🖥️ COMMAND LINE INTERFACE
```

---

## 6. Slash Commands System (`agent/commands.py`)

Nanobot intercepts messages starting with `/` as **system commands** that bypass the LLM entirely.

- **Purpose:** Fast, deterministic operations (model switching, session management) without AI inference.
- **Architecture:** `CommandHandler` uses a decorator-based registration pattern.
- **Session Switching:** Supports session overrides that persist across messages.

### Built-in Commands

| Command             | Description                              |
| ------------------- | ---------------------------------------- |
| `/help`             | List all available commands              |
| `/model`            | Show current model                       |
| `/model list`       | Fetch available models from provider API |
| `/model set <name>` | Switch to a different model              |
| `/session`          | Show current session info                |
| `/session <key>`    | Switch to a different session            |
| `/sessions`         | List all sessions                        |
| `/new`              | Create and switch to a new session       |
| `/context`          | Show context usage                       |
| `/context details`  | Show detailed context breakdown          |
| `/skills`           | List available skills                    |
| `/tools`            | List registered tools                    |
| `/clear`            | Clear current session history            |

### Adding Custom Commands

````python
@handler.register("mycommand")
def cmd_mycommand(args: list[str], ctx: dict[str, Any]) -> CommandResult:
    """My command description."""
    return CommandResult("Response content")
```

---

## 7. Subagents (`agent/subagent.py`)

Subagents are lightweight background agents spawned by the main agent for parallel task execution.

### Lifecycle
1. Main agent spawns subagent with a task
2. Subagent runs independently (max 15 iterations)
3. Subagent reports result back via message bus
4. Subagent terminates

### Configurable Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | string | The task to complete (required) |
| `label` | string | Display name |
| `model` | string | LLM model override (e.g., `gpt-4o-mini` for cost savings) |
| `system_prompt` | string | Custom instructions |
| `tools` | array | Tools to enable |

### Available Tools for Subagents

`read_file`, `write_file`, `list_dir`, `exec_shell`, `web_search`, `web_fetch`, `browser`

### Usage Example

```json
{
  "task": "Check website for updates",
  "model": "gpt-4o-mini",
  "system_prompt": "Only report notable changes",
  "tools": ["browser", "web_fetch"]
}
```

---

## 8. How to Extend Nanobot

### Adding a New Tool

1.  Create a class inheriting from `BaseTool`.
2.  Implement `name`, `description`, and `run()`.
3.  Register it in `AgentLoop._register_default_tools` (or dynamically).

### Adding a New Channel

1.  Create `nanobot/channels/mychannel.py`.
2.  Inherit from `BaseChannel`.
3.  Implement `start()`, `stop()`, and `send()`.
4.  In `start()`, set up your listener (e.g., webhook or polling) to call `self.bus.publish_inbound()`.
5.  Add to `ChannelManager._init_channels`.

### Adding a New Skill

1.  Create `skills/my_new_skill/SKILL.md`.
2.  Add YAML frontmatter describing the skill.
3.  Write clear Markdown instructions on how the agent should use the underlying tools (like `exec_shell`) to accomplish the task.

---

## 9. Configuration Schema Deep Dive

The configuration is managed via Pydantic models in `nanobot/config/schema.py`. It is crucial to respect the structure defined here when editing `~/.nanobot/config.json`.

- **Environment Variables:** Env vars prefixed with `NANOBOT_` can override config files (e.g., `NANOBOT_PROVIDERS__OPENAI__API_KEY` overrides `providers.openai.apiKey`).
- **Hidden Security Flags:**
  - `tools.restrict_to_workspace` (Default: `false`): If set to `true`, **ALL** file system operations (read/write/list) and shell execution are strictly confined to the workspace directory. This is critical for public or shared deployments to prevent the agent from accessing sensitive system files.
  - `channels.{name}.allow_from` (Default: `[]`): An empty list allows **anyone** to interact with the bot on that channel. To secure the bot, populate this list with trusted user IDs (e.g., specific Telegram IDs or Discord snowflakes).

---

## 10. The "Bootstrap" System

The agent's personality and instructions are NOT hardcoded in Python. They are loaded from special Markdown files in the workspace root. This allows for rapid iteration on the agent's behavior without touching code.

- `IDENTITY.md`: Defines _who_ the agent is (Name, Role, Tone).
- `AGENTS.md`: High-level directives and behavioral guidelines.
- `SOUL.md`: Deeper philosophical or core personality traits.
- `TOOLS.md`: Custom instructions on tool usage patterns.
- `USER.md`: Information about the user (preferences, bio) that the agent should always know.

**Mechanism:** `ContextBuilder._load_bootstrap_files` iterates through this list and appends content directly to the system prompt if the files exist.

---

## 11. Context Construction Lifecycle

Understanding _what_ the LLM sees is key to debugging. The `ContextBuilder` (`agent/context.py`) assembles the prompt in this specific order:

1.  **Identity:** Time, Runtime environment, Workspace path.
2.  **Bootstrap Files:** `IDENTITY.md` → `AGENTS.md` → `SOUL.md` etc.
3.  **Memory:** Content from `memory/MEMORY.md` (Short/Medium term knowledge).
4.  **Active Skills:** Full content of skills marked as "always active".
5.  **Skills Summary:** List of _available_ skills (name + description) so the agent knows what it _can_ load.
6.  **Conversation History:** The actual chat log.
7.  **Current Session:** Metadata about the current channel and user (e.g., "Channel: telegram, Chat ID: 12345").

This structure ensures the agent prioritizes its core instructions (Identity) while having access to dynamic capabilities (Skills) and context (History).

---

## 12. Security Boundaries

Security is enforced at two layers:

1.  **Ingress Layer (Authentication):**
    - The `ChannelManager` checks `allow_from` lists _before_ passing messages to the bus. Unauthorized users are ignored or rejected at the gate.
2.  **Execution Layer (Sandboxing):**
    - When `restrict_to_workspace` is enabled, the `AgentLoop` initializes `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `ListDirTool`, and `ExecTool` with a `restrict_to_workspace=True` flag.
    - These tools internally validate paths before execution, raising errors if a path attempts to escape the workspace (e.g., `../../etc/passwd`).

---

## 13. Multi-Modal Architecture

Nanobot is natively multi-modal (text + vision).

- **Ingestion:** The `InboundMessage` class supports a `media` field, which is a list of file paths.
- **Processing:**
  - When an image is received (e.g., via WhatsApp), it is downloaded to a temporary location.
  - `ContextBuilder` detects these media paths.
  - It converts images to **Base64** strings.
  - It constructs a multi-modal message payload (mixing `text` and `image_url` types) compliant with OpenAI/Anthropic vision APIs.
- **Implication:** You can show the agent a screenshot or photo, and it will "see" it as part of the conversation history.

---

## 14. Tool Execution Logging

Every tool execution is logged to session-specific files for audit and verification.

- **Location:** `<workspace>/logs/<session_key>.jsonl`
- **Format:** JSON Lines (one JSON object per line)
- **Content:** Each entry contains:
  - `timestamp`: ISO 8601 UTC timestamp
  - `tool`: Name of the executed tool
  - `params`: Parameters passed to the tool
  - `result`: Execution result (truncated to 2000 chars)
  - `duration_ms`: Execution time in milliseconds

**Implementation:** The `ToolLogger` class (`nanobot/agent/tools/tool_logger.py`) is initialized by `AgentLoop` and passed to `ToolRegistry.execute()` calls.

---

## 15. Critical Implementation Details

- **AsyncIO:** The entire core is async. Do not use blocking calls in the main loop.
- **Configuration:** `~/.nanobot/config.json` is the source of truth. Config schema is in `nanobot/config/schema.py` (Pydantic).
- **State:** Conversation state is stored in `nanobot/session/`. It is file-based for simplicity.
- **Subagents:** The `SpawnTool` allows the agent to create isolated task runners. These share the bus but have their own context.

## 16. Development & Debugging

- **Logs:** Uses `loguru`. Logs are printed to stderr.
- **Running locally:**
  ```bash
  python -m nanobot agent -m "Hello"
````

- **Testing:**
  - Use `pytest` for unit tests.
  - Test channels using the `nanobot gateway` command.
