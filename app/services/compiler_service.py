from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

from app.models import CompileDiagnostic, CompositionSpec, ValidationResult


@dataclass
class CompileResult:
    ok: bool
    diagnostics: list[CompileDiagnostic]


class CompilerService:
    def compile(self, tsx_code: str, spec: CompositionSpec, asset_paths: list[str]) -> CompileResult:
        diagnostics: list[CompileDiagnostic] = []
        if tsx_code.count("{") != tsx_code.count("}"):
            diagnostics.append(CompileDiagnostic(category="syntax", message="Unbalanced braces in generated TSX."))
        if tsx_code.count("<AbsoluteFill") < 1:
            diagnostics.append(CompileDiagnostic(category="structure", message="Missing root AbsoluteFill component."))
        for scene in spec.scenes:
            # Check that the image_id is referenced AND that at least one real path appears in the TSX.
            if scene.image_id not in tsx_code:
                diagnostics.append(CompileDiagnostic(category="asset", message=f"Missing scene asset reference: {scene.image_id}"))
        # Verify that the assetPaths map is present when asset_paths were supplied.
        if asset_paths and "assetPaths" not in tsx_code:
            diagnostics.append(CompileDiagnostic(category="asset", message="assetPaths map is missing from generated TSX."))
        if not re.search(r"export const Composition", tsx_code):
            diagnostics.append(CompileDiagnostic(category="export", message="Composition export is missing."))
        return CompileResult(ok=not diagnostics, diagnostics=diagnostics)

    def classify(self, diagnostics: list[CompileDiagnostic]) -> str:
        categories = {d.category for d in diagnostics}
        if "asset" in categories:
            return "missing_asset"
        if "structure" in categories:
            return "missing_structure"
        if "syntax" in categories:
            return "syntax_error"
        return "unknown"

