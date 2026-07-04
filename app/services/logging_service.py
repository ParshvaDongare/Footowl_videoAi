from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RunLogger:
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def log(self, node: str, event: str, **payload: Any) -> None:
        self.events.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "node": node,
                "event": event,
                **payload,
            }
        )

