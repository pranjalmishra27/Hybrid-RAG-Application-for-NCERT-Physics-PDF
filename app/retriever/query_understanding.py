"""
Query Understanding Module
Classifies queries and routes retrieval strategy accordingly.

Different query types need different retrieval emphasis:
  - Definition  → BM25 heavy (exact term match) + vector
  - Formula     → BM25 heavy (formula notation) + graph (law→formula path)
  - Numerical   → vector + BM25 (numbers, units)
  - Comparison  → vector heavy (semantic similarity across chunks)
  - Conceptual  → vector + graph (concept relationships)
  - Multi-hop   → graph heavy (multi-hop paths) + vector
"""

import re
from typing import Dict, Tuple
from enum import Enum


class QueryType(str, Enum):
    DEFINITION = "definition"
    FORMULA = "formula"
    NUMERICAL = "numerical"
    COMPARISON = "comparison"
    CONCEPTUAL = "conceptual"
    MULTI_HOP = "multi_hop"


# Retrieval weight profiles per query type
# (vector_weight, bm25_weight, graph_weight) — relative emphasis
RETRIEVAL_PROFILES: Dict[QueryType, Dict] = {
    QueryType.DEFINITION: {
        "vector": 0.4,
        "bm25": 0.5,
        "graph": 0.1,
        "top_k_multiplier": 1.0,
        "rationale": "Definitions need exact term matching (BM25 strength)",
    },
    QueryType.FORMULA: {
        "vector": 0.3,
        "bm25": 0.5,
        "graph": 0.2,
        "top_k_multiplier": 1.2,
        "rationale": "Formulas need exact notation match + graph law→formula paths",
    },
    QueryType.NUMERICAL: {
        "vector": 0.4,
        "bm25": 0.5,
        "graph": 0.1,
        "top_k_multiplier": 1.0,
        "rationale": "Numerical questions need exact values and units (BM25)",
    },
    QueryType.COMPARISON: {
        "vector": 0.6,
        "bm25": 0.2,
        "graph": 0.2,
        "top_k_multiplier": 1.5,
        "rationale": "Comparisons need semantic similarity across different chunks",
    },
    QueryType.CONCEPTUAL: {
        "vector": 0.5,
        "bm25": 0.2,
        "graph": 0.3,
        "top_k_multiplier": 1.2,
        "rationale": "Conceptual questions benefit from graph concept relationships",
    },
    QueryType.MULTI_HOP: {
        "vector": 0.3,
        "bm25": 0.2,
        "graph": 0.5,
        "top_k_multiplier": 1.5,
        "rationale": "Multi-hop needs graph traversal across concept relationships",
    },
}

# Classification signal patterns
SIGNALS: Dict[QueryType, list] = {
    QueryType.DEFINITION: [
        r"\bwhat is\b", r"\bdefine\b", r"\bdefinition\b", r"\bmeaning of\b",
        r"\bwhat are\b", r"\bwhat does .+ mean\b", r"\bstate the meaning\b",
    ],
    QueryType.FORMULA: [
        r"\bformula\b", r"\bequation\b", r"\bexpress(ion)?\b", r"\bderive\b",
        r"\bderivation\b", r"\bmathematical(ly)?\b", r"\bwrite the .+ law\b",
        r"\bstate .+ (law|theorem)\b", r"\bgive the formula\b",
    ],
    QueryType.NUMERICAL: [
        r"\bcalculate\b", r"\bfind\b", r"\bcompute\b", r"\bdetermine\b",
        r"\bevaluate\b", r"\bgiven that\b", r"\bhow much\b", r"\bhow many\b",
        r"\bvalue of\b", r"\bnumerical\b", r"\bif .+ then\b",
    ],
    QueryType.COMPARISON: [
        r"\bdifference between\b", r"\bcompare\b", r"\bvs\b", r"\bversus\b",
        r"\bsimilar(ities)?\b", r"\bdistinguish\b", r"\bcontrast\b",
        r"\bboth\b .+ \band\b", r"\brelation between\b", r"\bdifferentiate\b",
    ],
    QueryType.MULTI_HOP: [
        r"\brelationship between\b", r"\bhow does .+ relate\b",
        r"\bwhy does .+ cause\b", r"\beffect of .+ on\b",
        r"\bconsequence\b", r"\blead to\b", r"\bresult in\b",
        r"\bhow (is|are) .+ connected\b",
    ],
    QueryType.CONCEPTUAL: [
        r"\bexplain\b", r"\bdescribe\b", r"\bhow does\b", r"\bwhy\b",
        r"\bwhat causes\b", r"\bprinciple\b", r"\bconcept\b",
        r"\bworking of\b", r"\bhow (do|does) .+ work\b",
    ],
}


def classify_query(query: str) -> Tuple[QueryType, Dict]:
    """
    Classify query and return (type, retrieval_profile).
    Uses weighted signal matching — multiple signals can fire.
    """
    q = query.lower().strip()
    scores = {qt: 0.0 for qt in QueryType}

    for qtype, patterns in SIGNALS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                # Comparison signals are more specific — give them double weight
                weight = 2.0 if qtype == QueryType.COMPARISON and "difference" in pattern else 1.0
                scores[qtype] += weight

    # Tiebreak heuristics
    # Physics formula indicator: contains = sign or known physics symbols
    if re.search(r'[A-Za-z]\s*=|∇|∮|∫|ε₀|μ₀|Φ|λ|ω', query):
        scores[QueryType.FORMULA] += 0.5

    # Multi-hop boost: contains two physics concepts
    physics_terms = ['charge', 'field', 'potential', 'current', 'resistance',
                     'magnetic', 'inductance', 'capacitance', 'wave', 'force', 'flux']
    term_count = sum(1 for t in physics_terms if t in q)
    if term_count >= 2:
        scores[QueryType.MULTI_HOP] += 0.3

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        best_type = QueryType.CONCEPTUAL  # safe default

    profile = RETRIEVAL_PROFILES[best_type]
    return best_type, profile


def expand_query(query: str, query_type: QueryType) -> str:
    """
    Query expansion for better retrieval coverage.
    Adds physics-specific synonyms and abbreviations.
    """
    expansions = {
        "coulomb": "Coulomb's law electric force F=kq1q2/r²",
        "gauss": "Gauss's law electric flux ε₀ surface integral",
        "faraday": "Faraday's law electromagnetic induction EMF flux",
        "lenz": "Lenz's law induced current opposes change",
        "ohm": "Ohm's law V=IR voltage current resistance",
        "kirchhoff": "Kirchhoff's laws KVL KCL voltage loop junction",
        "biot savart": "Biot-Savart law magnetic field current element",
        "ampere": "Ampere's law magnetic field current enclosed",
        "maxwell": "Maxwell's equations displacement current electromagnetic waves",
        "solenoid": "solenoid toroid magnetic field coil inductance",
        "capacitor": "capacitor capacitance charge voltage energy stored",
        "inductor": "inductor self-inductance mutual inductance flux linkage",
        "transformer": "transformer mutual induction primary secondary coil turns ratio",
    }

    expanded = query
    q_lower = query.lower()
    for key, expansion in expansions.items():
        if key in q_lower:
            expanded = f"{query} {expansion}"
            break

    # For formula queries, add math notation cues
    if query_type == QueryType.FORMULA:
        expanded += " formula equation mathematical expression"

    return expanded


def get_query_summary(query: str) -> Dict:
    """Full query analysis for the explainability panel."""
    qtype, profile = classify_query(query)
    expanded = expand_query(query, qtype)
    return {
        "original_query": query,
        "query_type": qtype.value,
        "retrieval_profile": profile,
        "expanded_query": expanded,
        "explanation": profile["rationale"],
    }
