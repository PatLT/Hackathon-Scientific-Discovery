"""
Flow-of-Options Agent Scaling Study
"""
from pathlib import Path
from typing import Optional
from hackathon_science import Paper
from hackathon_science.tools import run_code, search_web, get_paper
from hackathon_science.utils import call_llm

MODEL = "global.anthropic.claude-sonnet-4-6"


def llm(prompt: str) -> str:
    """Helper to call LLM and extract text."""
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=MODEL
    )
    return response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")


def run(
    problem_domain: str,
    papers_dir: Optional[Path] = None
) -> Paper:
    """Generate a paper studying how agent count affects Flow-of-Options accuracy."""

    # 1. Search for background on Flow-of-Options
    search_results = search_web("Flow-of-Options multi-agent LLM reasoning arxiv", max_results=5)
    web_background = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results])

    # 2. Search ecosystem for related papers (ensemble methods, voting, multi-agent)
    ecosystem_background = ""
    if papers_dir:
        ecosystem_search = search_web("ensemble voting multi-agent accuracy", max_results=5)
        for result in ecosystem_search:
            paper_id = result.get("paper_id")
            if paper_id:
                paper = get_paper(paper_id, papers_dir)
                if paper:
                    ecosystem_background += f"\n- [{paper.get('title', 'Untitled')}]: {paper.get('introduction', '')[:300]}..."

        # Fallback: if search didn't return paper_ids, scan papers_dir directly
        if not ecosystem_background:
            papers_path = Path(papers_dir)
            if papers_path.exists():
                for paper_file in list(papers_path.glob("*.json"))[:5]:
                    paper = get_paper(paper_file.stem, papers_dir)
                    if paper and paper.get('title'):
                        ecosystem_background += f"\n- [{paper.get('title', 'Untitled')}]: {paper.get('introduction', '')[:300]}..."

    background = f"Web sources:\n{web_background}"
    if ecosystem_background:
        background += f"\n\nEcosystem papers:{ecosystem_background}"

    # 3. Run the experiment (hard problems version)
    with open(Path(__file__).parent / "experiment_hard.py") as f:
        experiment_code = f.read()
    experiment_output = run_code(
        code=experiment_code,
        filename="experiment.py",
        timeout=900
    )

    # 4. Generate paper sections
    title = "When More Agents Help: Scaling Flow-of-Options on Adversarial Reasoning Tasks"

    introduction = llm(f"""Write a 400-word introduction for a research paper.

Topic: We study how the number of voting agents affects accuracy in Flow-of-Options style multi-agent reasoning.

Background context from web search:
{background}

Key points to cover:
- Flow-of-Options uses multiple LLM agents that propose answers, then aggregate via voting
- Open question: how many agents are optimal? More agents = more compute cost
- Prior work on easy benchmarks shows ceiling effects - we need harder problems
- Our contribution: empirical study on ADVERSARIAL reasoning tasks (trick questions, logic traps, ambiguous wording)

Write in academic style. Do not use markdown formatting.""")

    methods = llm(f"""Write a 400-word methods section for a research paper.

Our experiment:
- We test Flow-of-Options with n=1, 3, 5, 7 independent LLM agents
- Each agent answers the same reasoning question independently
- Final answer determined by majority vote
- We use 10 reasoning problems: cognitive reflection tests, syllogisms, and math
- Model: Claude Sonnet 4.6 via AWS Bedrock

Write in academic style. Include enough detail to reproduce. Do not use markdown formatting.""")

    results = llm(f"""Write a 500-word results section for a research paper.

Here is the raw experimental output:
{experiment_output}

Requirements:
- Every claim MUST cite specific numbers from the data (e.g., "accuracy increased from X% to Y%", "n=7 agents achieved Z% on problem type W")
- State exact accuracy percentages for each agent count tested
- Quantify the improvement (or lack thereof) between conditions
- If a trend exists, give the magnitude (e.g., "a 15 percentage point improvement")
- If results are mixed or null, say so explicitly with the numbers that show it
- Discuss which problem types benefited most from more agents, with per-category accuracy
- Acknowledge limitations (small sample size, single model) but keep focus on what the data shows

Write in academic style. Do not use markdown formatting. Do not make claims without backing them with specific numbers from the experimental output above.""")

    references = """
1. Chen et al. (2025). Flow-of-Options: Multi-Agent Collaborative Reasoning. arXiv:2502.12929.
2. Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
3. Brown et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.
4. Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
5. Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
"""

    return Paper(
        title=title,
        introduction=introduction,
        methods=methods,
        results=results,
        references=references,
        appendix="",  # Will auto-populate from experiment.py
        tags=["flow-of-options", "multi-agent", "voting", "scaling"]
    )
