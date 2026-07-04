from __future__ import annotations

from app.models import CompositionSpec
from app.services.ai_service import AIService


def build_tsx(spec: CompositionSpec, ai: AIService) -> str:
    return ai.generate_tsx(spec)

