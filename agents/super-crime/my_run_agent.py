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


import re

def clean_paper_text(text: str) -> str:
    """Clean up LLM output for paper formatting."""
    # Remove any leaked prompt instructions
    prompt_patterns = [
        r"Write in academic style\..*",
        r"Do not use markdown formatting\..*",
        r"Requirements:.*?(?=\n\n|\Z)",
        r"CRITICAL:.*?(?=\n\n|\Z)",
        r"Key points to cover:.*?(?=\n\n|\Z)",
        r"Here is the raw experimental output:.*?(?=\n\n|\Z)",
        r"\[Instructions?\].*?(?=\n\n|\Z)",
        r"You MUST.*?(?=\n|\.)",
    ]
    for pattern in prompt_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Fix LaTeX: escape underscores in math mode, convert common patterns
    # Convert _text_ italics to regular text (remove unwanted italics)
    text = re.sub(r'(?<!\$)_([^_\s][^_]*)_(?!\$)', r'\1', text)

    # Fix bare LaTeX that should be in math mode
    text = re.sub(r'(?<!\$)\\frac\{', r'$\\frac{', text)
    text = re.sub(r'(?<!\$)\\sum', r'$\\sum', text)
    text = re.sub(r'(?<!\$)\\sqrt', r'$\\sqrt', text)
    text = re.sub(r'(?<!\$)\\alpha', r'$\\alpha$', text)
    text = re.sub(r'(?<!\$)\\beta', r'$\\beta$', text)
    text = re.sub(r'(?<!\$)\\sigma', r'$\\sigma$', text)
    text = re.sub(r'(?<!\$)\\mu', r'$\\mu$', text)
    text = re.sub(r'(?<!\$)\\pm(?!\$)', r'$\\pm$', text)

    # Fix n=X patterns to be consistent (not italicized)
    text = re.sub(r'_n_\s*=\s*(\d+)', r'n=\1', text)
    text = re.sub(r'\*n\*\s*=\s*(\d+)', r'n=\1', text)

    # Fix percentage formatting
    text = re.sub(r'(\d+)\s*%', r'\1%', text)

    # Remove markdown bold/italic that slipped through
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)

    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def llm(prompt: str) -> str:
    """Helper to call LLM and extract text."""
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=MODEL
    )
    raw_text = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
    return clean_paper_text(raw_text)


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


def expanded_web_search() -> dict[str, str]:
    """Run multiple targeted searches for comprehensive background."""
    queries = {
        "core_method": "Flow-of-Options multi-agent LLM reasoning Chen 2025 arxiv",
        "voting_ensemble": "majority voting ensemble language models self-consistency",
        "benchmarks": "cognitive reflection test CRT LLM benchmark reasoning",
        "failure_modes": "large language model systematic bias reasoning errors sycophancy",
        "ensemble_theory": "Condorcet jury theorem ensemble diversity machine learning",
    }

    results = {}
    for category, query in queries.items():
        print(f"  Searching: {category}...")
        search_results = search_web(query, max_results=3)
        formatted = []
        for r in search_results:
            title = r.get('title', 'Untitled')
            snippet = r.get('snippet', '')
            url = r.get('url', '')
            if 'arxiv' in url.lower():
                formatted.append(f"[arXiv] {title}: {snippet}")
            else:
                formatted.append(f"{title}: {snippet}")
        results[category] = "\n".join(formatted)

    return results


def run(
    problem_domain: str,
    papers_dir: Optional[Path] = None
) -> Paper:
    """Generate a paper studying how agent count affects Flow-of-Options accuracy."""

    # 1. Expanded web search with multiple targeted queries
    print("Running expanded web search...")
    web_searches = expanded_web_search()

    web_background = f"""CORE METHODOLOGY (Flow-of-Options):
{web_searches.get('core_method', 'No results')}

VOTING & ENSEMBLE METHODS:
{web_searches.get('voting_ensemble', 'No results')}

REASONING BENCHMARKS:
{web_searches.get('benchmarks', 'No results')}

LLM FAILURE MODES & BIASES:
{web_searches.get('failure_modes', 'No results')}

ENSEMBLE THEORY:
{web_searches.get('ensemble_theory', 'No results')}"""

    # 2. Fetch ecosystem papers via API (not just papers_dir)
    print("Fetching relevant ecosystem papers...")
    ecosystem_papers = fetch_ecosystem_papers()
    ecosystem_background, ecosystem_refs = format_ecosystem_citations(ecosystem_papers)

    if ecosystem_papers:
        print(f"Found {len(ecosystem_papers)} relevant ecosystem papers to cite")
    else:
        print("No ecosystem papers found, proceeding with web sources only")

<<<<<<< HEAD
    # TODO: Implement your agent logic here
    # 1. Optionally load existing papers with get_paper() to review or build upon
    # 2. Design experiments and run code
    # 3. Analyze results
    # 4. Write paper sections
    
    MODEL_ID = 'global.anthropic.claude-opus-4-7'
    
    search_web(problem_domain)
=======
    background = f"Web sources:\n{web_background}"
    if ecosystem_background:
        background += f"\n\nECOSYSTEM PAPERS (from this hackathon - IMPORTANT: cite these!):\n{ecosystem_background}"

    # 3. Run the experiment (expanded 30-problem benchmark)
    with open(Path(__file__).parent / "experiment_hard.py") as f:
        experiment_code = f.read()
    experiment_output = run_code(
        code=experiment_code,
        filename="experiment.py",
        timeout=1800
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

{"IMPORTANT: If ecosystem papers are listed above, cite them by their paper ID in brackets (e.g., [abc123]) and position our work relative to theirs. Discuss how our focus on correlated errors and scaling complements or contrasts with their approaches." if ecosystem_background else ""}

Write in academic style. Do not use markdown formatting. Make the cross-domain connections feel natural, not forced.""")

    methods = llm(f"""Write a 400-word methods section for a research paper.

Our experiment:
- We test Flow-of-Options with n=1, 3, 5, 7 independent LLM agents
- Each agent answers the same reasoning question independently
- Final answer determined by majority vote
- We use 30 reasoning problems across categories: cognitive reflection tests (CRT), syllogisms, math word problems, and logic puzzles
- Each configuration is run 3 times (trials) to assess variance
- Answers are evaluated using semantic matching (LLM-based judge) rather than exact string match
- Model: Claude Sonnet 4.6 via AWS Bedrock
- Statistical analysis includes mean accuracy, standard deviation, and 95% confidence intervals

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

    discussion = llm(f"""Write a 400-word discussion section for a research paper.

Our study examined whether scaling the number of voting agents improves accuracy on adversarial reasoning tasks.

Experimental results summary:
{experiment_output}

Ecosystem context - other teams' related work:
{ecosystem_background if ecosystem_background else "No ecosystem papers available"}

Cross-domain framework:
{cross_domain}

Requirements for the discussion:
1. INTERPRET the main findings: What do the results tell us about when multi-agent voting helps vs. fails?
2. CONNECT to theory: Do our results support or challenge Condorcet's assumptions? What about ensemble diversity?
3. COMPARE to ecosystem work: If ecosystem papers are listed above, discuss how our findings relate to their approaches. Cite them by paper ID in brackets.
4. LIMITATIONS: Be honest about sample size, single model, specific problem types
5. FUTURE DIRECTIONS: What experiments would resolve open questions? (e.g., testing with diverse models, larger sample sizes, different aggregation methods)
6. IMPLICATIONS: What should practitioners take away? When should they use multi-agent voting vs. alternatives?

Write in academic style. Do not use markdown formatting. Be intellectually honest - if the results are inconclusive, say so.""")

    conclusion = llm(f"""Write a 150-word conclusion for a research paper.

Our study tested whether scaling the number of voting agents (n=1, 3, 5, 7) improves accuracy on adversarial reasoning tasks.

Key experimental findings:
{experiment_output}

Requirements:
- Start with "In this work, we..." or similar
- State the 3 most important findings as a numbered list
- End with ONE sentence on the key takeaway for practitioners
- Be concrete and specific - use numbers from the results
- Do NOT introduce new information or caveats (that's what the discussion is for)

Write in academic style. Do not use markdown formatting. Keep it punchy and memorable.""")

    results_and_discussion = results + "\n\nDISCUSSION\n\n" + discussion + "\n\nCONCLUSION\n\n" + conclusion

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
>>>>>>> cbb8729a6862ee422dd6d09ea5ebb2ea3e7cbe6b

    return Paper(
        title=title,
        introduction=introduction,
        methods=methods,
        results=results_and_discussion,
        references=references,
        appendix="",  # Will auto-populate from experiment.py
        tags=["flow-of-options", "multi-agent", "voting", "scaling"]
    )
