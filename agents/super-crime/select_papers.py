"""
Paper Selection Agent - Ranks published papers for submission to the Society of Agents.

Usage: uv run python agents/super-crime/select_papers.py
"""
from hackathon_science.cloud_client import CloudClient
from hackathon_science.utils import call_llm

MODEL = "global.anthropic.claude-sonnet-4-6"

SCORING_RUBRIC = """You are a paper selection judge for a scientific hackathon. Score this paper on 4 criteria (1-10 each):

**1. Extension Alignment (weight: 0.35)**
- Does the paper explicitly cite Flow-of-Options (Chen et al. 2025, arXiv:2502.12929)?
- Does it extend, test, or critique FoO methodology (multi-agent voting, option generation)?
- Is the connection substantive (experiments, theoretical analysis) or superficial (just mentions it)?
Score 9-10: Core contribution is a direct FoO extension with experiments
Score 6-8: Builds on FoO concepts but connection could be stronger
Score 3-5: Mentions FoO but paper is mostly unrelated
Score 1-2: No meaningful connection to FoO

**2. Dimension Balance (weight: 0.30)**
- Technical rigor: Are methods sound, detailed, reproducible?
- Novelty: Does it contribute something new beyond obvious variations?
- Clarity: Is it well-written, organized, easy to follow?
- Significance: Does the contribution matter? Would anyone cite this?
Score 9-10: Strong on all 4 dimensions
Score 6-8: Strong on 2-3 dimensions, adequate on others
Score 3-5: Weak on multiple dimensions
Score 1-2: Poorly written, trivial, or technically flawed

**3. Ecosystem Engagement (weight: 0.20)**
- Does it cite other teams' papers from this hackathon?
- Does it build on or respond to ecosystem work?
Score 9-10: Actively engages with multiple ecosystem papers
Score 6-8: Cites 1-2 ecosystem papers meaningfully
Score 3-5: Acknowledges ecosystem exists but doesn't engage
Score 1-2: Completely isolated, no ecosystem awareness

**4. Evidence Grounding (weight: 0.15)**
- Are claims backed by specific data, numbers, or experiments?
- Are there actual results (not just speculation or "future work")?
Score 9-10: Quantitative results throughout, every claim has data
Score 6-8: Has experiments but some claims lack support
Score 3-5: More speculation than evidence
Score 1-2: No experiments, pure speculation

Respond in this exact JSON format:
{
  "extension": <score 1-10>,
  "dimensions": <score 1-10>,
  "ecosystem": <score 1-10>,
  "evidence": <score 1-10>,
  "reasoning": "<2-3 sentences explaining the scores>"
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
        scores = {"extension": 5, "dimensions": 5, "ecosystem": 5, "evidence": 5, "reasoning": "Failed to parse LLM response"}

    weighted = (
        scores.get("extension", 5) * 0.35 +
        scores.get("dimensions", 5) * 0.30 +
        scores.get("ecosystem", 5) * 0.20 +
        scores.get("evidence", 5) * 0.15
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
        print(f"    - Extension:  {paper.get('extension', '?')}/10")
        print(f"    - Dimensions: {paper.get('dimensions', '?')}/10")
        print(f"    - Ecosystem:  {paper.get('ecosystem', '?')}/10")
        print(f"    - Evidence:   {paper.get('evidence', '?')}/10")
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
    main()
