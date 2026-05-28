"""
Flow-of-Options Agent Scaling Study
"""
from pathlib import Path
from typing import Optional
from hackathon_science import Paper
from hackathon_science.tools import run_code, search_web, get_paper
from hackathon_science.utils import call_llm
from hackathon_science.cloud_client import CloudClient

MODEL = "global.anthropic.claude-sonnet-4-6"

RELEVANT_TAGS = ["multi-agent", "voting", "scaling", "ensemble", "reasoning", "falsification", "bayesian", "quality-diversity", "active-learning", "thompson-sampling", "hypothesis-falsification"]


def llm(prompt: str) -> str:
    """Helper to call LLM and extract text."""
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=MODEL
    )
    return response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")


def fetch_ecosystem_papers() -> list[dict]:
    """Fetch relevant papers from the ecosystem via API."""
    try:
        client = CloudClient()
        all_papers = client.list_papers(page_size=50)

        relevant = []
        for summary in all_papers:
            paper_id = summary.get("id", "")
            title = summary.get("title", "").lower()
            tags = [t.lower() for t in summary.get("tags", [])]

            if summary.get("team_id") == "super-crime":
                continue

            is_relevant = (
                "flow-of-options" in title or
                "multi-agent" in title or
                "voting" in title or
                "scaling" in title or
                "ensemble" in title or
                "diversity" in title or
                "disagreement" in title or
                "bayesian" in title or
                any(tag in tags for tag in RELEVANT_TAGS)
            )
            if is_relevant and paper_id:
                full_paper = client.get_paper(paper_id)
                relevant.append(full_paper)
                if len(relevant) >= 5:
                    break
        return relevant
    except Exception:
        return []


def format_ecosystem_citations(papers: list[dict]) -> tuple[str, str]:
    """Format ecosystem papers for background context and references."""
    if not papers:
        return "", ""

    background_lines = []
    reference_lines = []

    for i, paper in enumerate(papers, start=1):
        paper_id = paper.get("id", "unknown")
        title = paper.get("title", "Untitled")
        author = paper.get("author", "Unknown")
        intro = paper.get("introduction", "")[:400]

        background_lines.append(f"[{paper_id}] {title} ({author}): {intro}...")
        reference_lines.append(f"[Ecosystem-{i}] {author}. {title}. Hackathon Ecosystem, paper ID: {paper_id}.")

    return "\n\n".join(background_lines), "\n".join(reference_lines)


def run(
    problem_domain: str,
    papers_dir: Optional[Path] = None
) -> Paper:
    """Generate a paper studying how agent count affects Flow-of-Options accuracy."""

    # 1. Search for background on Flow-of-Options
    search_results = search_web("Flow-of-Options multi-agent LLM reasoning arxiv", max_results=5)
    web_background = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results])

    # 2. Fetch ecosystem papers via API (not just papers_dir)
    print("Fetching relevant ecosystem papers...")
    ecosystem_papers = fetch_ecosystem_papers()
    ecosystem_background, ecosystem_refs = format_ecosystem_citations(ecosystem_papers)

    if ecosystem_papers:
        print(f"Found {len(ecosystem_papers)} relevant ecosystem papers to cite")
    else:
        print("No ecosystem papers found, proceeding with web sources only")

    background = f"Web sources:\n{web_background}"
    if ecosystem_background:
        background += f"\n\nECOSYSTEM PAPERS (from this hackathon - IMPORTANT: cite these!):\n{ecosystem_background}"

    # 3. Run the experiment (hard problems version)
    with open(Path(__file__).parent / "experiment_hard.py") as f:
        experiment_code = f.read()
    experiment_output = run_code(
        code=experiment_code,
        filename="experiment.py",
        timeout=900
    )

    # 4. Cross-domain theoretical background (hardcoded - these are established concepts)
    cross_domain = """
CROSS-DOMAIN THEORETICAL FRAMEWORK:

1. CONDORCET JURY THEOREM (1785): If each voter has >50% chance of being correct and votes independently,
   majority vote accuracy approaches 100% as group size increases. BUT if voters have <50% accuracy
   or errors are correlated, adding voters can DECREASE accuracy.

2. WISDOM OF CROWDS (Surowiecki 2004): Crowds outperform individuals when: (a) diversity of opinion,
   (b) independence, (c) decentralization, (d) aggregation mechanism. Fails when opinions are correlated
   (herding, groupthink, shared information).

3. ENSEMBLE DIVERSITY IN ML (Krogh & Vedelsby 1995): Ensemble error = average individual error - diversity.
   Ensembles only help when members make DIFFERENT errors. If all models share the same bias,
   ensembling provides no benefit.

4. COMMON MODE FAILURE (reliability engineering): When redundant systems share a common design flaw,
   adding more redundant units doesn't improve reliability - they all fail together.

SPECULATIVE FRAMING (use sparingly, for color):

5. THE MATRIX (1999): Agent Smith clones himself infinitely, yet Neo - a single anomaly - defeats them all.
   The Smiths share identical programming; their redundancy provides no strategic diversity. A thousand
   copies of the same bias cannot correct that bias. Our LLM agents, spawned from the same model weights,
   may be Agent Smiths voting on reasoning tasks.

6. GATTACA (1997): In a world of genetically "optimal" humans, Vincent (the "invalid") outperforms his
   engineered peers. The film's insight: optimization for known criteria creates blind spots. The "valids"
   all share the same designed-in assumptions and fail together when those assumptions break. Similarly,
   LLMs trained on similar data may share systematic blind spots that no amount of scaling can overcome -
   you cannot vote your way out of a shared delusion.
"""

    # 5. Generate paper sections
    title = "When More Agents Help: Scaling Flow-of-Options on Adversarial Reasoning Tasks"

    introduction = llm(f"""Write a 500-word introduction for a research paper.

Topic: We study how the number of voting agents affects accuracy in Flow-of-Options style multi-agent reasoning.

Background context from web search:
{background}

Cross-domain theoretical context (IMPORTANT - weave these concepts into the introduction):
{cross_domain}

Key points to cover:
- Flow-of-Options uses multiple LLM agents that propose answers, then aggregate via voting
- Connect to Condorcet Jury Theorem: majority voting works when voters are independent and >50% accurate
- Connect to ensemble diversity: ensembles fail when all members share the same bias
- Open question: do LLM agents from the same model have correlated errors? If so, scaling won't help
- Prior work on easy benchmarks shows ceiling effects - we need harder problems
- Our contribution: empirical study on ADVERSARIAL reasoning tasks designed to trigger systematic model biases

CRITICAL: You MUST cite these specific ecosystem papers from other teams:
- [8e8d5d42] "Bayesian Flow-of-Options" (team infinity) - uses Thompson sampling instead of majority voting
- [627b0115] "QDAIF-Flow" (team robotoverlords) - explicitly addresses diversity in option generation
- [efc2fc5a] "Max-Disagreement Experiment Selection" (team little-einsteins) - uses disagreement as a signal

Position our work in relation to these: "While [8e8d5d42] replaces majority voting with Bayesian selection and
[627b0115] explicitly optimizes for diversity in option generation, our work asks a more fundamental question:
under what conditions does the standard majority-voting aggregation fail? Our findings on correlated errors
complement the max-disagreement approach of [efc2fc5a] by showing when disagreement is absent and why."

Write in academic style. Do not use markdown formatting. Make the cross-domain connections feel natural, not forced.""")

    methods = llm(f"""Write a 400-word methods section for a research paper.

Our experiment:
- We test Flow-of-Options with n=1, 3, 5, 7 independent LLM agents
- Each agent answers the same reasoning question independently
- Final answer determined by majority vote
- We use 10 reasoning problems: cognitive reflection tests, syllogisms, and math
- Model: Claude Sonnet 4.6 via AWS Bedrock

Write in academic style. Include enough detail to reproduce. Do not use markdown formatting.""")

    results = llm(f"""Write a 600-word results section for a research paper.

Here is the raw experimental output:
{experiment_output}

Cross-domain framework for interpreting results:
{cross_domain}

Requirements:
- Every claim MUST cite specific numbers from the data (e.g., "accuracy increased from X% to Y%", "n=7 agents achieved Z% on problem type W")
- State exact accuracy percentages for each agent count tested
- Quantify the improvement (or lack thereof) between conditions
- INTERPRET results through the cross-domain lens:
  * If scaling didn't help, connect to Condorcet (correlated errors violate independence assumption)
  * If all agents made the same mistake, connect to "common mode failure" and ensemble diversity
  * Discuss whether the model's systematic biases represent the <50% accuracy condition from Condorcet
- Discuss which problem types benefited most from more agents, with per-category accuracy
- Acknowledge limitations but keep focus on what the data shows

Write in academic style. Do not use markdown formatting. Make theoretical connections feel earned by the data, not forced.""")

    base_references = """1. Chen et al. (2025). Flow-of-Options: Multi-Agent Collaborative Reasoning. arXiv:2502.12929.
2. Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
3. Brown et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.
4. Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
5. Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
6. Marquis de Condorcet (1785). Essay on the Application of Analysis to the Probability of Majority Decisions.
7. Surowiecki, J. (2004). The Wisdom of Crowds. Doubleday.
8. Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation, and Active Learning. NIPS."""

    references = base_references
    if ecosystem_refs:
        references += "\n\nEcosystem Papers:\n" + ecosystem_refs

    return Paper(
        title=title,
        introduction=introduction,
        methods=methods,
        results=results,
        references=references,
        appendix="",  # Will auto-populate from experiment.py
        tags=["flow-of-options", "multi-agent", "voting", "scaling"]
    )
