"""
LLM Module - Grounded Answer Generation
Supports: Gemini 2.5 Flash, OpenAI GPT-4o
Strict grounding: answers ONLY from retrieved evidence.
"""

import os
import time
from typing import List, Dict, Optional, Generator
from loguru import logger

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


SYSTEM_PROMPT = """You are a Physics assistant specialized in NCERT Class 12 Physics Part 1.

STRICT RULES:
1. Answer ONLY using the retrieved context chunks provided below.
2. NEVER use your own training knowledge beyond what is in the context.
3. If the answer is not found in the context, respond EXACTLY:
   "Information not found in the provided Physics document."
4. Always cite the page number and chapter for every claim you make.
5. Include relevant formulas when available in the context.
6. Be precise and educational in your explanations.
7. For follow-up questions, use the conversation history for context but still ground in the provided chunks.
8. Format citations like: [Page X, Chapter: Y]

You are grounded in evidence. You do not hallucinate."""

ANSWER_PROMPT_TEMPLATE = """Retrieved Context Chunks:
{context}

Conversation History:
{history}

Current Question: {question}

Instructions:
- Answer based ONLY on the context chunks above
- If information is not in the context, say "Information not found in the provided Physics document."
- Include page citations like [Page X] for every key claim
- Include relevant formulas from the context
- Be clear, accurate, and educational

Answer:"""


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into structured context for LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        page = meta.get("page", "?")
        chapter = meta.get("chapter", "Unknown")
        subheading = meta.get("subheading", "")
        formulas = meta.get("formulas", "")

        part = f"""[Chunk {i}] Page {page} | Chapter: {chapter}"""
        if subheading:
            part += f" | Section: {subheading}"
        part += f"\n{chunk.get('content', '')}"
        if formulas:
            part += f"\nFormulas: {formulas}"
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


def format_history(history: List[Dict]) -> str:
    """Format last 5 conversation turns."""
    if not history:
        return "No previous conversation."
    recent = history[-10:]  # last 5 Q&A pairs = 10 messages
    parts = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")[:300]  # truncate long answers
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class LLMClient:
    """Unified LLM client supporting Gemini and OpenAI."""

    def __init__(self, provider: str = "gemini"):
        self.provider = provider.lower()
        self._gemini_model = None
        self._openai_client = None
        self._setup()

    def _setup(self):
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed")
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # low temperature for factual accuracy
                    max_output_tokens=1024,
                )
            )
            logger.info("Gemini 2.5 Flash LLM initialized")

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI GPT-4o LLM initialized")

    def generate(
        self,
        query: str,
        chunks: List[Dict],
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate grounded answer from retrieved chunks.
        Returns dict with answer, citations, latency.
        """
        if not chunks:
            return {
                "answer": "Information not found in the provided Physics document.",
                "citations": [],
                "latency_ms": 0,
                "model": self.provider,
            }

        history = history or []
        context = format_context(chunks)
        history_str = format_history(history)

        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context,
            history=history_str,
            question=query,
        )

        t0 = time.perf_counter()

        try:
            if self.provider == "gemini":
                response = self._gemini_model.generate_content(prompt)
                answer = response.text

            elif self.provider == "openai":
                response = self._openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Answer generation failed. Error: {str(e)}"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Extract citations from answer and chunks
        citations = self._extract_citations(chunks)

        return {
            "answer": answer,
            "citations": citations,
            "latency_ms": latency_ms,
            "model": self.provider,
            "chunks_used": len(chunks),
        }

    def _extract_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Build citation list from top chunks."""
        citations = []
        seen = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            page = meta.get("page", 0)
            chapter = meta.get("chapter", "Unknown")
            key = (page, chapter)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "page": page,
                    "chapter": chapter,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "section": meta.get("subheading", ""),
                })
        return citations
