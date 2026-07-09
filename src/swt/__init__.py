"""SWT (Save The World) -- Odysseus's automated generator/critic/analyzer loop.

An automated loop shaped like ASI-EVOLVE's learn -> design -> experiment ->
analyze cycle: a generator answers, a critic finds problems, an analyzer
diagnoses the root cause and prescribes a typed adjustment, and every
diagnosis lands in a queryable experience store that seeds future runs.
Knowledge-base retrieval feeds the generator and is widened when the analyzer
diagnoses a retrieval failure; diagnosed knowledge gaps are written back so
they are known next time. A per-user preference model (learned from
accept/reject feedback) predicts whether the user -- not just the critic --
will accept the answer, and RLM-style recursive condensing keeps long
reference material from rotting the context.

Each role (generator / critic / analyzer / librarian) takes a full Odysseus
model spec, so ``model@endpoint`` routes roles to different nodes in a
distributed deployment; bare names keep everything on one box.

Public surface:
    LoopConfig, LoopResult, RoundResult   -- request/result schemas
    Prescription, RetrievalAdjustment     -- the analyzer's typed levers
    OdysseusModelAdapter                  -- routes calls through Odysseus's LLM
    ExperienceStore, SwtStore, create_store -- pluggable persistence
    Retriever, RagRetriever               -- knowledge-base retrieval seam
    PreferenceModel                       -- user accept/reject predictor
    run_loop                              -- the engine (async event generator)
"""
from src.swt.engine import run_loop
from src.swt.models import ModelAdapter, OdysseusModelAdapter
from src.swt.preference import PreferenceModel, default_embed_fn
from src.swt.retrieval import KnowledgeWriter, RagRetriever, Retriever
from src.swt.schemas import (
    Critique,
    Diagnosis,
    DiagnosisCategory,
    LoopConfig,
    LoopResult,
    Prescription,
    RetrievalAdjustment,
    RoundResult,
    SatisfactionPrediction,
)
from src.swt.store import ExperienceStore, SwtStore, create_store

# Deprecated alias for the renamed preference model.
CognitiveModel = PreferenceModel

__all__ = [
    "CognitiveModel",
    "PreferenceModel",
    "default_embed_fn",
    "run_loop",
    "ModelAdapter",
    "OdysseusModelAdapter",
    "Critique",
    "Diagnosis",
    "DiagnosisCategory",
    "LoopConfig",
    "LoopResult",
    "Prescription",
    "RetrievalAdjustment",
    "RoundResult",
    "SatisfactionPrediction",
    "ExperienceStore",
    "SwtStore",
    "create_store",
    "Retriever",
    "KnowledgeWriter",
    "RagRetriever",
]
