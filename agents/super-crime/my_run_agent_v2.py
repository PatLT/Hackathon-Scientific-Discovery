"""
Flow-of-Options Scaling Study - Clean Version
"""
from pathlib import Path
from typing import Optional
from hackathon_science import Paper
from hackathon_science.tools import run_code
from hackathon_science.utils import call_llm

MODEL = "global.anthropic.claude-sonnet-4-6"


def llm(prompt: str) -> str:
    """Call LLM and return text."""
    resp = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=MODEL
    )
    return resp.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")


def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    """Generate a paper on multi-agent voting scaling."""

    # 1. Run experiment (30 problems, 3 trials, harder items)
    print("Running experiment...")
    with open(Path(__file__).parent / "experiment_v2.py") as f:
        code = f.read()
    experiment_output = run_code(code=code, filename="experiment.py", timeout=2400)
    print("Experiment complete.")

    # 2. Generate sections with real references baked in
    title = "Does More Mean Better? Scaling Multi-Agent Voting in Flow-of-Options Reasoning"

    intro_prompt = f"""Write a 500-word introduction for an academic paper.

TOPIC: We empirically test whether increasing the number of voting agents improves accuracy in Flow-of-Options multi-agent reasoning.

BACKGROUND TO INCLUDE:
- Flow-of-Options (Chen et al., 2025) uses multiple LLM agents that independently propose answers, then aggregates via majority vote
- Condorcet's Jury Theorem (1785): majority vote converges to truth IF voters are >50% accurate AND independent
- Ensemble diversity (Krogh & Vedelsby, 1995): ensemble error = mean error - diversity. No diversity = no benefit.
- Self-consistency (Wang et al., 2023) showed voting over chain-of-thought paths improves reasoning
- Key question: when agents come from the SAME model, are errors correlated? If so, Condorcet fails.

STYLE: Academic, no markdown, no citations in brackets like [abc123]. Use author-date format (Smith, 2020)."""

    introduction = llm(intro_prompt)

    methods_prompt = f"""Write a 350-word methods section for an academic paper.

OUR EXPERIMENT:
- Tested Flow-of-Options with n=1, 3, 5, 7 agents
- 30 reasoning problems across 5 categories: arithmetic (6), logic (6), trick questions (9), probability/base-rate (6), counterfactual (3)
- 3 independent trials per condition for statistical robustness
- All agents used Claude Sonnet 4.6 via AWS Bedrock
- Final answer by majority vote
- Flexible semantic matching for answer evaluation
- Report mean accuracy with standard deviation and 95% confidence intervals

STYLE: Academic, reproducible detail, no markdown."""

    methods = llm(methods_prompt)

    results_prompt = f"""Write a 500-word results section for an academic paper.

EXPERIMENTAL DATA:
{experiment_output}

REQUIREMENTS:
- Report mean accuracy ± standard deviation for each agent count
- Include 95% confidence intervals where available
- Compare baseline (n=1) to maximum (n=7) with effect size
- Break down by category: which improved, which didn't, which got worse
- Identify common mode failures (problems ALL agents got wrong across ALL conditions)
- Be precise: use exact numbers from the data

STYLE: Academic, data-driven, no markdown."""

    results = llm(results_prompt)

    discussion_prompt = f"""Write a 300-word discussion for an academic paper.

KEY FINDINGS:
{experiment_output}

INTERPRET THROUGH THEORY:
- If scaling didn't help: agents share biases (violates Condorcet independence)
- If all agents made same error: common mode failure (like redundant systems with shared design flaw)
- Ensemble diversity principle: same model = zero diversity = no ensemble benefit
- Implications for practitioners: when to use multi-agent voting, when not to

STYLE: Academic, thoughtful, no markdown."""

    discussion = llm(discussion_prompt)

    # Fixed references - real papers only
    references = """
1. Chen, L., et al. (2025). Flow-of-Options: Diversified Reasoning with Multi-Agent Language Models. arXiv:2502.12929.
2. Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
3. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
4. Brown, T., et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.
5. Krogh, A. & Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation, and Active Learning. NIPS 1995.
6. de Condorcet, M. (1785). Essay on the Application of Analysis to the Probability of Majority Decisions.
7. Surowiecki, J. (2004). The Wisdom of Crowds. Doubleday.
8. Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
9. Frederick, S. (2005). Cognitive Reflection and Decision Making. Journal of Economic Perspectives, 19(4), 25-42.
"""

    return Paper(
        title=title,
        introduction=introduction,
        methods=methods,
        results=results + "\n\n" + discussion,
        references=references,
        appendix="",
        tags=["flow-of-options", "multi-agent", "voting", "ensemble", "scaling"]
    )
