"""LLM-as-a-judge evaluation prompts for faithfulness and citation verification."""

# Faithfulness Judge Prompt
# Evaluates whether the claims in the generated answer are strictly supported
# by the provided retrieved context chunks, scoring from 1 to 5.
FAITHFULNESS_JUDGE_PROMPT = """You are an impartial, expert AI judge evaluating the faithfulness of a RAG (Retrieval-Augmented Generation) answer against retrieved source contexts.

Retrieved Context:
{context}

User Question:
{question}

Generated Answer:
{answer}

Task:
Evaluate whether every claim made in the Generated Answer is directly and factually supported by the Retrieved Context.
- Score 5: Fully faithful. Every single claim is explicitly supported by the context.
- Score 4: Mostly faithful. Minor extrapolations that are logically obvious and non-contradictory.
- Score 3: Partially faithful. Contains some claims supported by context and some unsubstantiated claims.
- Score 2: Mostly unfaithful. Significant hallucinations or contradictions to the context.
- Score 1: Completely unfaithful or hallucinated.

If the Generated Answer is a refusal stating that the context lacks sufficient information (e.g., contains NO_ANSWER or a refusal statement) when the context indeed lacks the answer, award a score of 5.

Output your judgment strictly in JSON format with two keys:
{{
  "reasoning": "brief 1-2 sentence explanation",
  "score": <integer from 1 to 5>
}}
"""

# Citation Verification Prompt
# Evaluates whether each cited chunk number [N] actually contains the specific claim it is cited for.
CITATION_VERIFIER_PROMPT = """You are an auditor verifying citation accuracy in a RAG response.

Claim with Citation:
"{claim}"

Cited Passage Content [{citation_number}]:
"{passage}"

Task:
Determine whether Cited Passage [{citation_number}] directly contains or supports the claim.

Output strictly valid JSON:
{{
  "supports_claim": <true or false>,
  "explanation": "brief 1 sentence reason"
}}
"""
