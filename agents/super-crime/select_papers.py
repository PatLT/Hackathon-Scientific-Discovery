"""
Paper Selection Agent - Ranks published papers for submission to the Society of Agents.

Usage: uv run python agents/super-crime/select_papers.py
"""
from hackathon_science.cloud_client import CloudClient
from hackathon_science.utils import call_llm

MODEL = "global.anthropic.claude-sonnet-4-6"

SCORING_RUBRIC = """You are a demanding peer reviewer for a scientific venue. Score this paper on 4 criteria (1-10 each), matching how real ML reviewers evaluate submissions:

**1. Technical Quality (weight: 0.40)**
This is THE most critical dimension. Reviewers consistently give 0/10 for:
- No statistical testing (p-values, confidence intervals, effect sizes)
- No repeated trials or variance analysis
- Small sample sizes without power analysis
- Missing ablations and sensitivity analysis
- Incomplete reproducibility details (seeds, hyperparameters, hardware)
- Unvalidated assumptions
- Evaluation artifacts that confound results
Score 9-10: Formal proofs OR rigorous experiments with statistical tests, confidence intervals, ablations, multiple baselines
Score 6-8: Solid methodology but missing some rigor (e.g., no confidence intervals)
Score 3-5: Experiments exist but lack statistical analysis, small N, limited baselines
Score 1-2: No statistical testing, tiny dataset, acknowledged confounds, unreproducible
Score 0: Major methodological flaw that invalidates conclusions

**2. Novelty (weight: 0.25)**
- Is this a new method, benchmark, theory, or substantial empirical finding?
- Or does it repackage known intuitions with a small probe?
- Is related work coverage comprehensive?
Score 9-10: Novel method or surprising empirical discovery, comprehensive related work
Score 6-8: Incremental but meaningful extension, decent positioning
Score 3-5: Repackages known ideas, narrow related work
Score 1-2: Well-known result, no new contribution

**3. Clarity (weight: 0.15)**
- Well-written, organized, easy to follow?
- Includes figures and tables (essential for empirical work)?
- Notation is consistent and defined?
Score 9-10: Excellent writing, useful visualizations, precise notation
Score 6-8: Readable but missing figures/tables or some imprecision
Score 3-5: Hard to follow, disorganized
Score 1-2: Poorly written, confusing

**4. Significance (weight: 0.20)**
- Important problem? Timely?
- Would this change how people think or work?
- Actionable method others could adopt?
Score 9-10: High-impact problem, demonstrates substantial advancement
Score 6-8: Relevant problem, useful but limited contribution
Score 3-5: Narrow interest, cautionary note rather than enabling result
Score 1-2: Trivial problem or no practical value

Be harsh like a real reviewer. Most hackathon papers score 0-1 on Technical Quality because they lack statistical rigor.

Respond in this exact JSON format:
{
  "technical": <score 0-10>,
  "novelty": <score 1-10>,
  "clarity": <score 1-10>,
  "significance": <score 1-10>,
  "reasoning": "<2-3 sentences explaining scores, especially Technical Quality weaknesses>"
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

    weighted = (
        scores.get("technical", 0) * 0.40 +
        scores.get("novelty", 3) * 0.25 +
        scores.get("clarity", 5) * 0.15 +
        scores.get("significance", 3) * 0.20
    )
    scores["weighted"] = round(weighted, 2)
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
        print(f"    Weighted Score: {paper['weighted']}/10")
        print(f"    - Technical:    {paper.get('technical', '?')}/10")
        print(f"    - Novelty:      {paper.get('novelty', '?')}/10")
        print(f"    - Clarity:      {paper.get('clarity', '?')}/10")
        print(f"    - Significance: {paper.get('significance', '?')}/10")
        print(f"    Reasoning: {paper.get('reasoning', 'N/A')}")

    if len(ranked) > 2:
        print("\n" + "-"*60)
        print("PAPERS NOT SELECTED")
        print("-"*60)
        for paper in ranked[2:]:
            print(f"\n- {paper['title']} (score: {paper['weighted']}/10)")
            print(f"  ID: {paper['paper_id']}")
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
