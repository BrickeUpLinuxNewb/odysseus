"""Back-compat shim: the "cognitive model" is now the preference model.

The original module here sold a lexical accept/reject heuristic as a
"TRIBE v2-inspired cognitive model". The mechanism now lives in
``src.swt.preference`` under its honest name; import
:class:`~src.swt.preference.PreferenceModel` directly in new code.
"""
from src.swt.preference import PreferenceModel

CognitiveModel = PreferenceModel

__all__ = ["CognitiveModel", "PreferenceModel"]
