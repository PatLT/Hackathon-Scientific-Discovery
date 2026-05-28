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
    """Clean up LLM output for paper formatting. Removes markdown, fixes LaTeX, strips prompt leaks."""

    # 1. Remove leaked prompt instructions (aggressive patterns)
    prompt_patterns = [
        r"Write in academic style\..*?(?=\n|$)",
        r"Do not use markdown formatting\..*?(?=\n|$)",
        r"Requirements:.*?(?=\n\n|\Z)",
        r"CRITICAL.*?(?=\n\n|\Z)",
        r"Key points to cover:.*?(?=\n\n|\Z)",
        r"Here is the raw experimental output:.*?(?=\n\n|\Z)",
        r"\[Instructions?\].*?(?=\n\n|\Z)",
        r"You MUST.*?(?=\n|\.)",
        r"IMPORTANT:.*?(?=\n|$)",
        r"Note:.*?(?=\n|$)",
        r"Remember to.*?(?=\n|$)",
        r"Make sure to.*?(?=\n|$)",
        r"Please ensure.*?(?=\n|$)",
    ]
    for pattern in prompt_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # 2. Remove ALL markdown formatting
    # Headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Italic (careful not to break LaTeX subscripts)
    text = re.sub(r'(?<![\\$])_([^_\s][^_]*)_(?![\\$])', r'\1', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)
    # Strikethrough
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Blockquotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Images ![alt](url) -> remove entirely
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    # Bullet points (convert to plain text)
    text = re.sub(r'^\s*[-*+]\s+', '- ', text, flags=re.MULTILINE)
    # Numbered lists with periods
    text = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)

    # 3. Fix/remove problematic LaTeX
    # Remove bare LaTeX commands that aren't in math mode (convert to plain text)
    latex_to_plain = {
        r'\\alpha': 'alpha',
        r'\\beta': 'beta',
        r'\\gamma': 'gamma',
        r'\\delta': 'delta',
        r'\\sigma': 'sigma',
        r'\\mu': 'mu',
        r'\\pi': 'pi',
        r'\\theta': 'theta',
        r'\\lambda': 'lambda',
        r'\\epsilon': 'epsilon',
        r'\\pm': '±',
        r'\\times': '×',
        r'\\div': '÷',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
        r'\\infty': '∞',
        r'\\sum': 'sum',
        r'\\prod': 'product',
        r'\\sqrt': 'sqrt',
        r'\\frac': '',
        r'\\left': '',
        r'\\right': '',
        r'\\text': '',
        r'\\textbf': '',
        r'\\textit': '',
        r'\\emph': '',
        r'\\cite': '',
        r'\\ref': '',
        r'\\label': '',
    }
    for latex, plain in latex_to_plain.items():
        # Only replace if NOT inside $...$
        text = re.sub(rf'(?<!\$){latex}(?!\$)', plain, text)

    # Remove stray $ signs that aren't paired
    # Count $ and if odd, they're broken - just remove all
    if text.count('$') % 2 != 0:
        text = text.replace('$', '')

    # Remove LaTeX curly braces that are orphaned
    text = re.sub(r'(?<!\\)\{([^{}]*)\}', r'\1', text)

    # 4. Fix common formatting issues
    # n=X patterns (normalize spacing)
    text = re.sub(r'_?n_?\s*=\s*(\d+)', r'n=\1', text)
    text = re.sub(r'\*n\*\s*=\s*(\d+)', r'n=\1', text)

    # Percentage formatting
    text = re.sub(r'(\d+)\s*%', r'\1%', text)

    # Fix doubled punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])(?=[A-Za-z])', r'\1 ', text)

    # 5. Clean up whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)  # Empty lines with whitespace
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
    title = "Scaling Multi-Agent Voting on Adversarial Reasoning: A Statistical Analysis of Flow-of-Options"

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

    methods = llm(f"""Write a 500-word methods section for a research paper. This section must be detailed enough for reproduction.

EXPERIMENTAL SETUP:
- We test Flow-of-Options with n=1, 3, 5, 7 independent LLM agents
- Each agent answers the same reasoning question independently
- Final answer determined by majority vote
- Model: Claude Sonnet 4.6 via AWS Bedrock (inference profile: global.anthropic.claude-sonnet-4-6)
- Temperature: default (API default)
- No system prompt; user prompt: "Answer with just one word or number, no explanation: [question]"

BENCHMARK:
- 30 reasoning problems across 6 categories (5 problems each):
  * Arithmetic: multi-digit multiplication, addition
  * Syllogistic logic: categorical syllogisms with novel terms
  * CRT-style trick questions: problems with intuitive but wrong answers
  * Semantic traps: questions with misleading framing (e.g., "Moses" instead of "Noah")
  * Counterfactual reasoning: hypothetical worlds with altered facts
  * Probability/base rate: questions testing base rate neglect, gambler's fallacy

EVALUATION PROTOCOL:
- Semantic matching via LLM judge (avoids brittle string comparison)
- Judge prompt asks whether answer is semantically equivalent considering numeric formats, synonyms
- Fallback to exact match for unambiguous cases (efficiency optimization)

STATISTICAL ANALYSIS:
- Each configuration run for 3 independent trials
- Report: mean accuracy, standard deviation across trials, 95% confidence intervals
- CI computed using t-distribution with df=2 (t-critical=4.303 for 95% CI)
- Effect size: Cohen's d for n=1 vs n=7 comparison
- Per-category breakdown with variance

BASELINES:
- Majority vote (primary method under study)
- Oracle upper bound: correct if ANY agent in ensemble got the answer right (shows aggregation ceiling)

Write in academic style with precise technical detail. Include the exact statistical formulas used. Do not use markdown formatting.""")

    results = llm(f"""Write a 700-word results section for a research paper. Structure it around the tables in the data.

Here is the raw experimental output (contains Tables 1-3 and Key Findings):
{experiment_output}

Cross-domain framework for interpreting results:
{cross_domain}

CRITICAL REQUIREMENTS:
1. REPRODUCE THE TABLES from the experimental output. Reference them as "Table 1", "Table 2", "Table 3".
2. For EVERY claim, cite specific numbers with confidence intervals where available.
   BAD: "Accuracy improved with more agents"
   GOOD: "Mean accuracy increased from 53.3% (95% CI: [48.2%, 58.4%]) at n=1 to 60.0% (95% CI: [55.1%, 64.9%]) at n=7"
3. Report the effect size (Cohen's d) and interpret it (negligible/small/medium/large).
4. Discuss the Oracle upper bound - what does the gap between majority vote and oracle tell us?
5. Break down results by category - which problem types benefited most/least from scaling?
6. Report standard deviations to show trial-to-trial variance.

INTERPRETATION (connect to theory, but only where data supports it):
- If effect size is small/negligible despite more agents, this suggests correlated errors (Condorcet violation)
- If oracle >> majority vote, the right answer exists but aggregation fails to surface it
- Per-category differences may reveal which problem types induce systematic bias

Write in academic style. Include the tables. Do not use markdown headers (use plain text section labels if needed). Every quantitative claim needs a number from the data.""")

    discussion = llm(f"""Write a 500-word discussion section for a research paper.

Our study examined whether scaling the number of voting agents improves accuracy on adversarial reasoning tasks.

Experimental results summary:
{experiment_output}

Ecosystem context - other teams' related work:
{ecosystem_background if ecosystem_background else "No ecosystem papers available"}

Cross-domain framework:
{cross_domain}

STRUCTURE YOUR DISCUSSION AS FOLLOWS:

1. MAIN FINDING (1 paragraph): State the primary result with effect size. Is the improvement statistically meaningful or negligible?

2. THEORETICAL INTERPRETATION (1 paragraph):
   - If scaling helped: Does this support Condorcet? What does it tell us about error independence?
   - If scaling didn't help: Does this suggest correlated errors? Common mode failure?
   - What does the oracle gap tell us about aggregation vs. generation?

3. CATEGORY-LEVEL INSIGHTS (1 paragraph): Which problem types responded best/worst to scaling? Why might this be?

4. LIMITATIONS (1 paragraph - be specific and quantitative):
   - Sample size: 30 problems, 3 trials (state exact numbers)
   - Single model (Claude Sonnet 4.6) - cannot generalize to other LLMs
   - Semantic matching via LLM judge may have its own biases
   - Agent count range (1-7) may not capture inflection points

5. FUTURE WORK (1 paragraph): Specific experiments that would address limitations:
   - Larger benchmark (100+ problems)
   - Multiple models (cross-model ensembles)
   - Alternative aggregation (weighted voting, learned aggregators)
   - Direct measurement of error correlation between agents

{"6. ECOSYSTEM CONTEXT: If papers are listed above, compare our findings to theirs. Cite by paper ID." if ecosystem_background else ""}

Write in academic style. Be honest about what the data does and doesn't show. Do not use markdown formatting.""")

    conclusion = llm(f"""Write a 200-word conclusion for a research paper.

Our study tested whether scaling the number of voting agents (n=1, 3, 5, 7) improves accuracy on adversarial reasoning tasks.

Key experimental findings:
{experiment_output}

Requirements:
- Start with "In this work, we..." or similar
- State the 3 most important findings with SPECIFIC NUMBERS:
  1. Overall effect of scaling (accuracy change from n=1 to n=7, with effect size)
  2. Best/worst performing category
  3. Gap between majority vote and oracle upper bound
- End with ONE sentence on the key takeaway for practitioners
- Do NOT introduce new information or caveats

EXAMPLE FORMAT (adapt numbers from actual data):
"In this work, we studied whether scaling... Our key findings: (1) Accuracy changed from X% to Y%, an effect size of Z (interpretation). (2) Category A showed the largest benefit while Category B showed no improvement. (3) The oracle upper bound of W% suggests that better aggregation methods could recover an additional V percentage points. For practitioners, this suggests..."

Write in academic style. Do not use markdown formatting.""")

    results_and_discussion = results + "\n\nDISCUSSION\n\n" + discussion + "\n\nCONCLUSION\n\n" + conclusion

    base_references = """1. Chen et al. (2025). Flow-of-Options: Multi-Agent Collaborative Reasoning. arXiv:2502.12929.
2. Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
3. Brown et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.
4. Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
5. Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
6. Marquis de Condorcet (1785). Essay on the Application of Analysis to the Probability of Majority Decisions.
7. Surowiecki, J. (2004). The Wisdom of Crowds. Doubleday.
8. Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation, and Active Learning. NIPS.
9. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum Associates.
10. Frederick, S. (2005). Cognitive Reflection and Decision Making. Journal of Economic Perspectives, 19(4), 25-42.
11. Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.
12. Liang et al. (2022). Holistic Evaluation of Language Models. arXiv:2211.09110."""

    references = base_references
    if ecosystem_refs:
        references += "\n\nEcosystem Papers:\n" + ecosystem_refs

    appendix = f"""APPENDIX A: EXPERIMENT CODE

The following Python code implements the Flow-of-Options scaling experiment described in the Methods section.

```python
{experiment_code}
```"""

    return Paper(
        title=title,
        introduction=introduction,
        methods=methods,
        results=results_and_discussion,
        references=references,
        appendix=appendix,
        tags=["flow-of-options", "multi-agent", "voting", "scaling"]
    )
