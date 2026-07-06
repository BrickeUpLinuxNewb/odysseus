"""SWT (Save The World) -- Odysseus's automated generator/critic/analyzer loop.

An automated micro-loop that maps to the ASI-EVOLVE framework: a generator
answers, a critic finds problems, and an analyzer diagnoses the root cause and
adjusts the next round -- closing the loop without a human. A TRIBE v2-inspired
cognitive model predicts whether the user (not just the critic) will accept the
answer, and RLM-style recursive condensing keeps long reference material from
rotting the context.

Public surface:
    LoopConfig, LoopResult, RoundResult   -- request/result schemas
    OdysseusModelAdapter                  -- routes calls through Odysseus's LLM
    SwtStore                              -- isolated SQLite persistence
    CognitiveModel                        -- user-satisfaction predictor
    run_loop                              -- the engine (async event generator)
"""
from src.swt.cognitive_model import CognitiveModel
from src.swt.engine import run_loop
from src.swt.models import ModelAdapter, OdysseusModelAdapter
from src.swt.schemas import (
    Critique,
    Diagnosis,
    DiagnosisCategory,
    LoopConfig,
    LoopResult,
    RoundResult,
    SatisfactionPrediction,
)
from src.swt.store import SwtStore

__all__ = [
    "CognitiveModel",
    "run_loop",
    "ModelAdapter",
    "OdysseusModelAdapter",
    "Critique",
    "Diagnosis",
    "DiagnosisCategory",
    "LoopConfig",
    "LoopResult",
    "RoundResult",
    "SatisfactionPrediction",
    "SwtStore",
]
