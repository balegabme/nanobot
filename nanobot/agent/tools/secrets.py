"""Secrets tool for storing and retrieving sensitive values."""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class SecretsTool(Tool):
    """
    Secure storage for API keys, wallet keys, and other secrets.
    
    Stores in ~/.nanobot/secrets.json (separate from config for easy .gitignore).
    """
    
    name = "secrets"
    description = """Manage secrets (API keys, wallet keys, etc). Actions:
- store: Save a secret (params: key, value)
- get: Retrieve a secret (params: key)
- list: List all secret keys (no values shown)
- delete: Remove a secret (params: key)"""
    
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "get", "list", "delete"],
                "description": "Action to perform"
            },
            "key": {
                "type": "string",
                "description": "Secret key name (e.g., ETH_WALLET_KEY)"
            },
            "value": {
                "type": "string",
                "description": "Secret value (for store action)"
            }
        },
        "required": ["action"]
    }
    
    def __init__(self, secrets_path: Path | None = None):
        self.secrets_path = secrets_path or Path.home() / ".nanobot" / "secrets.json"
    
    def _load(self) -> dict[str, str]:
        """Load secrets from file."""
        if not self.secrets_path.exists():
            return {}
        try:
            return json.loads(self.secrets_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load secrets: {e}")
            return {}
    
    def _save(self, secrets: dict[str, str]) -> None:
        """Save secrets to file."""
        self.secrets_path.parent.mkdir(parents=True, exist_ok=True)
        self.secrets_path.write_text(
            json.dumps(secrets, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    async def execute(
        self,
        action: str,
        key: str | None = None,
        value: str | None = None,
        **kwargs: Any
    ) -> str:
        if action == "store":
            return self._store(key, value)
        elif action == "get":
            return self._get(key)
        elif action == "list":
            return self._list()
        elif action == "delete":
            return self._delete(key)
        return f"Unknown action: {action}"
    
    def _store(self, key: str | None, value: str | None) -> str:
        if not key or not value:
            return "Error: key and value are required"
        
        secrets = self._load()
        secrets[key] = value
        self._save(secrets)
        
        logger.info(f"Stored secret: {key}")
        return f"Secret '{key}' stored successfully"
    
    def _get(self, key: str | None) -> str:
        if not key:
            return "Error: key is required"
        
        secrets = self._load()
        if key not in secrets:
            return f"Secret '{key}' not found"
        
        return secrets[key]
    
    def _list(self) -> str:
        secrets = self._load()
        if not secrets:
            return "No secrets stored"
        
        keys = list(secrets.keys())
        return f"Stored secrets ({len(keys)}): " + ", ".join(keys)
    
    def _delete(self, key: str | None) -> str:
        if not key:
            return "Error: key is required"
        
        secrets = self._load()
        if key not in secrets:
            return f"Secret '{key}' not found"
        
        del secrets[key]
        self._save(secrets)
        
        logger.info(f"Deleted secret: {key}")
        return f"Secret '{key}' deleted"
