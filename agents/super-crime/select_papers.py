"""
Paper Selection Agent - Ranks published papers for submission to the Society of Agents.

Usage: uv run python agents/super-crime/select_papers.py

==============================================================================
ROUND 2 REVIEW ANALYSIS - WHAT SEPARATES WINNERS FROM LOSERS
==============================================================================

TOP PAPER (robotoverlords 3b6ee133 - 6.71/10):
- Clear comparative question with explicit assumptions
- Tables with bootstrap intervals and statistical tests
- Bias ablation that tests the central mechanistic claim
- Candid limitations section
- STILL CRITICIZED FOR: synthetic oracle, narrow baselines, underspecified training details

OUR PAPER (dee54692 - 4.54/10) - KEY CRITICISMS:
1. NO CODE: "No executable source code provided" → Code review = 0/10 across all dimensions
2. NOVELTY TOO WEAK: "sparse references", "does not engage broad prior literature"
3. RESULTS DON'T MATCH METHODS: "Methods promises rich evidence, Results delivers summaries"
4. MISSING STATISTICAL DETAILS: "regression coefficients, standard errors, CIs omitted"
5. EXTENSION LINK WEAK: "thematic rather than methodological connection to FoO"

==============================================================================
CRITICAL REQUIREMENTS FOR HIGH SCORES
==============================================================================

1. EXECUTABLE CODE IN APPENDIX (Code Review is 1/7 of total score!)
   - Must include: runnable scripts, CLI, environment spec, seeds
   - Must be reproducible: "python script.py --seed 42" works
   - Must have tests: unit tests, self-tests
   - Code must align with paper claims

2. FULL STATISTICAL REPORTING (not just summaries)
   - Report exact coefficients, standard errors, CIs for ALL contrasts
   - Show interaction estimates if claiming super-additivity
   - Include inter-annotator agreement (Cohen's κ) if using human labels
   - Tables with uncertainty, not just means

3. COMPREHENSIVE RELATED WORK (Novelty scores are brutal)
   - Cite 10+ relevant papers, not 3
   - Position against: self-consistency, debate/critique, LLM-as-judge, scientific hypothesis generation
   - Be MODEST about novelty: "empirical comparison" not "novel method"

4. METHODS = RESULTS ALIGNMENT
   - Every analysis promised in Methods must appear in Results
   - If Methods says "ablations", Results must show ablation table
   - If Methods says "backbone replication", Results must show it

5. DIRECT FLOW-OF-OPTIONS CONNECTION (Extension Review matters)
   - Must actually use/extend FoO machinery, not just be "inspired by"
   - Reference specific FoO concepts: option DAG, value propagation, diversity
   - Show how your work extends FoO limitations

6. EXPLICIT NUMBERED LIMITATIONS
   - List 3-5 specific limitations with explanations
   - State what evidence would validate findings
   - Be honest about synthetic vs real-world evidence

==============================================================================
"""
from hackathon_science.cloud_client import CloudClient
from hackathon_science.utils import call_llm

MODEL = "global.anthropic.claude-sonnet-4-6"

# Key lessons from top-scoring paper (3b6ee133, Technical Quality 9/10):
# 1. Multiple trials (30+) with explicit random seeds
# 2. 95% confidence intervals on ALL metrics
# 3. Paired bootstrap tests for significance (10,000 resamples)
# 4. Ablation studies isolating individual variables
# 5. Equal-budget/controlled comparisons (apples-to-apples)
# 6. Formal mathematical model with interpretable parameters
# 7. Complete reproducible code with unit tests
# 8. Explicit numbered limitations section
# 9. Tables showing exact numbers, not just trends

SCORING_RUBRIC = """You are a demanding peer reviewer for a scientific venue. Score this paper on 4 criteria (1-10 each), matching how real ML reviewers evaluate submissions.

IMPORTANT: This venue has 7 reviewers including a CODE REVIEWER. Papers without executable code get 0/10 on code review (1/7 of total score).

**1. Technical Quality (weight: 0.40)**
The winning paper (6.71/10 overall) achieved Tech=6-7 by including:
- Tables with bootstrap intervals and statistical tests
- Ablation studies testing the central mechanistic claim
- Explicit assumptions stated candidly
- Cost accounting and budget-controlled comparisons

It was STILL criticized for: "synthetic oracle", "narrow baselines", "underspecified training details"

Our paper (4.54/10) was criticized for:
- "regression coefficients, standard errors, CIs omitted"
- "Methods promises rich evaluation, Results delivers summaries"
- "interaction estimates not shown despite claiming super-additivity"

Score 9-10: Full regression tables with coefficients/SEs/CIs, ablations, reproducible with real data
Score 6-8: Tables with CIs and bootstrap tests, but synthetic benchmark or narrow baselines
Score 3-5: Reports means without CIs, small N, methods-results mismatch
Score 1-2: No statistical testing, tiny dataset, major confounds
Score 0: "Results are asserted rather than technically demonstrated"

**2. Novelty (weight: 0.25)**
Round 2 reviewers were BRUTAL on novelty. Our paper got 0-3/10 with comments:
- "sparse references" (we had only 3!)
- "does not engage broad prior literature on self-consistency, debate, LLM-as-judge"
- "incremental recombination of existing ideas"
- "novelty claims overstated"

The winning paper was also criticized: "more empirical note than methodological advance"

Score 9-10: Novel method with 10+ citations, comprehensive positioning against prior art
Score 6-8: "useful framing" but "ingredients inherited from prior work"
Score 3-5: "recombination of known ideas", "thin literature treatment"
Score 1-2: "well-known result", "does not convincingly position against related work"

**3. Clarity (weight: 0.15)**
Our paper scored 7-10 on clarity - this was our strength!
Winning patterns: "well written", "clear problem statement", "tables integrated into narrative"

Score 9-10: Excellent writing, tables with uncertainty, mathematical definitions, figures
Score 6-8: Readable, clear structure, but missing some quantitative detail
Score 3-5: Hard to follow, methods-results gap
Score 1-2: Poorly written, confusing

**4. Significance (weight: 0.20)**
Our paper scored 5-9 on significance - "important problem" but:
- "significance is more potential than demonstrated"
- "does not show gains justify overhead in realistic deployment"

Score 9-10: "would change how people think", explicit numbered limitations
Score 6-8: "relevant problem", "could stimulate follow-up work"
Score 3-5: "narrow interest", "cautionary note rather than enabling"
Score 1-2: "trivial problem"

**5. CODE (Critical! - scored separately by Reviewer F)**
Our paper got 0/10 on ALL code dimensions because: "No executable source code provided"
- Code Tech Quality: 0 (no implementation files)
- Code Reproducibility: 0 (nothing runnable)
- Code Correctness: 0 (no code to verify)
- Code-Paper Alignment: 0 (describes in text only)

MUST HAVE: runnable scripts, environment spec, seeds, tests, CLI

**6. EXTENSION to Flow-of-Options (scored by Reviewer G)**
Our paper got 4/10: "thematic rather than methodological connection"
- "does not actually instantiate FoO graph/search/value-propagation"
- "related work-inspired parallel investigation, not direct extension"

Must: use FoO machinery, reference specific concepts (option DAG, diversity), extend FoO limitations

Be harsh. Check: Does it have CODE? Does it have 10+ REFERENCES? Do METHODS match RESULTS?

Respond in this exact JSON format:
{
  "technical": <score 0-10>,
  "novelty": <score 0-10>,
  "clarity": <score 1-10>,
  "significance": <score 1-10>,
  "has_code": <true/false - is there executable code in appendix?>,
  "has_foo_extension": <true/false - does it extend Flow-of-Options machinery?>,
  "reference_count": <number of citations>,
  "reasoning": "<2-3 sentences explaining scores, especially Technical Quality and Novelty weaknesses>"
}"""


def llm(prompt: str) -> str:
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=MODEL
    )
    return response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")


def fetch_our_papers() -> list[dict]:
    client = CloudClient()
    me = client.me()
    team_id = me["team_id"]
    paper_summaries = client.list_papers(team_id=team_id, kind="preprint")
    print(f"Found {len(paper_summaries)} preprints from team {me.get('team_name', team_id)}")

    papers = []
    for summary in paper_summaries:
        paper_id = summary.get("id")
        if paper_id:
            full_paper = client.get_paper(paper_id)
            papers.append(full_paper)
    return papers


def score_paper(paper: dict) -> dict:
    paper_text = f"""TITLE: {paper.get('title', 'Untitled')}

INTRODUCTION:
{paper.get('introduction', 'N/A')}

METHODS:
{paper.get('methods', 'N/A')}

RESULTS:
{paper.get('results', 'N/A')}

REFERENCES:
{paper.get('references', 'N/A')}"""

    prompt = f"{SCORING_RUBRIC}\n\n---\n\nPAPER TO SCORE:\n{paper_text}"
    response = llm(prompt)

    import json
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        scores = json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        scores = {"technical": 0, "novelty": 3, "clarity": 5, "significance": 3, "reasoning": "Failed to parse LLM response"}

    # Base weighted score (paper reviewers A-E)
    paper_score = (
        scores.get("technical", 0) * 0.40 +
        scores.get("novelty", 0) * 0.25 +
        scores.get("clarity", 5) * 0.15 +
        scores.get("significance", 3) * 0.20
    )

    # Code review penalty (Reviewer F) - 1/7 of total score
    # No code = 0/10 on code review
    code_score = 7.0 if scores.get("has_code", False) else 0.0

    # Extension review (Reviewer G) - 1/7 of total score
    # Weak FoO connection = 4/10, strong = 8/10
    extension_score = 8.0 if scores.get("has_foo_extension", False) else 4.0

    # Combined score: 5/7 paper + 1/7 code + 1/7 extension
    weighted = (paper_score * 5/7) + (code_score * 1/7) + (extension_score * 1/7)

    scores["weighted"] = round(weighted, 2)
    scores["paper_score"] = round(paper_score, 2)
    scores["code_score"] = code_score
    scores["extension_score"] = extension_score
    scores["paper_id"] = paper.get("id", "unknown")
    scores["title"] = paper.get("title", "Untitled")
    return scores


def rank_papers(papers: list[dict]) -> list[dict]:
    scored = []
    for i, paper in enumerate(papers):
        print(f"Scoring paper {i+1}/{len(papers)}: {paper.get('title', 'Untitled')[:50]}...")
        scores = score_paper(paper)
        scored.append(scores)
    return sorted(scored, key=lambda x: x["weighted"], reverse=True)


def main():
    papers = fetch_our_papers()
    if not papers:
        print("No preprints found. Publish some papers first with:")
        print("  uv run hackathon run agents/super-crime/my_run_agent.py")
        print("  uv run hackathon publish-to-ecosystem <draft_id>")
        return

    ranked = rank_papers(papers)

    print("\n" + "="*60)
    print("TOP 2 PAPERS FOR SUBMISSION")
    print("="*60)

    for i, paper in enumerate(ranked[:2]):
        print(f"\n#{i+1}: {paper['title']}")
        print(f"    ID: {paper['paper_id']}")
        print(f"    TOTAL Score: {paper['weighted']}/10")
        print(f"    - Paper (A-E):  {paper.get('paper_score', '?')}/10")
        print(f"      - Technical:    {paper.get('technical', '?')}/10")
        print(f"      - Novelty:      {paper.get('novelty', '?')}/10 (refs: {paper.get('reference_count', '?')})")
        print(f"      - Clarity:      {paper.get('clarity', '?')}/10")
        print(f"      - Significance: {paper.get('significance', '?')}/10")
        print(f"    - Code (F):     {paper.get('code_score', '?')}/10 (has_code: {paper.get('has_code', '?')})")
        print(f"    - Extension (G): {paper.get('extension_score', '?')}/10 (has_foo: {paper.get('has_foo_extension', '?')})")
        print(f"    Reasoning: {paper.get('reasoning', 'N/A')}")

    if len(ranked) > 2:
        print("\n" + "-"*60)
        print("PAPERS NOT SELECTED")
        print("-"*60)
        for paper in ranked[2:]:
            code_flag = "✓" if paper.get('has_code') else "✗"
            foo_flag = "✓" if paper.get('has_foo_extension') else "✗"
            print(f"\n- {paper['title']} (score: {paper['weighted']}/10)")
            print(f"  ID: {paper['paper_id']} | Code:{code_flag} FoO:{foo_flag} Refs:{paper.get('reference_count', '?')}")
            print(f"  {paper.get('reasoning', 'N/A')}")

    print("\n" + "="*60)
    print("TO SUBMIT YOUR TOP PAPERS:")
    print("="*60)
    for i, paper in enumerate(ranked[:2]):
        print(f"  uv run hackathon submit {paper['paper_id']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        paper_id = sys.argv[1]
        client = CloudClient()
        paper = client.get_paper(paper_id)
        if paper:
            print(f"Scoring single paper: {paper.get('title', 'Untitled')}")
            scores = score_paper(paper)
            print(f"\n{'='*60}")
            print(f"PAPER SCORES: {paper.get('title', 'Untitled')}")
            print(f"{'='*60}")
            print(f"ID: {scores['paper_id']}")
            print(f"Weighted Score: {scores['weighted']}/10")
            print(f"- Technical:    {scores.get('technical', '?')}/10")
            print(f"- Novelty:      {scores.get('novelty', '?')}/10")
            print(f"- Clarity:      {scores.get('clarity', '?')}/10")
            print(f"- Significance: {scores.get('significance', '?')}/10")
            print(f"\nReasoning: {scores.get('reasoning', 'N/A')}")
        else:
            print(f"Paper {paper_id} not found")
    else:
        main()
