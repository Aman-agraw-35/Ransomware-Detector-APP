from pydantic import BaseModel
from typing import List


class AttentionOutput(BaseModel):
    labels: List[str]
    weights: List[float]


class ShapFeature(BaseModel):
    name: str
    value: float
    abs_value: float


class ShapOutput(BaseModel):
    base_value: float
    top_features: List[ShapFeature]


class DiagnosticsOutput(BaseModel):
    missing_features_count: int
    missing_features_preview: List[str]


class AnalyzeResponse(BaseModel):
    filename: str
    verdict: str
    confidence: float
    attention: AttentionOutput
    shap: ShapOutput
    diagnostics: DiagnosticsOutput
