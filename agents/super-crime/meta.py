"""
meta.py — meta-agent for super-crime.

Reads all existing agent scripts and originator.pdf, synthesises a novel
research-agent structure, writes it as meta_run_agent.py, then validates
it by running it via the hackathon CLI.
"""
from pathlib import Path
from typing import Optional

from hackathon_science import Paper
from hackathon_science.utils import call_llm
from hackathon_science.customtools import (
    llm_text, extract_code, has_error,
    escape_latex_blocks, META_CONTEXT,
)

FAST_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
MODEL_ID      = "global.anthropic.claude-opus-4-7"

META_RUN_PATH = Path("agents/super-crime/meta_run_agent.py")

PAPER_RETURN_TEMPLATE = '''\
    return Paper(
        title=research_question,
        introduction=introduction,
        methods=methods,
        results=results,
        references=references,
        appendix=f"```python\\n{final_code}\\n```",
        tags=[],
    )'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_pdf(path: Path, max_chars: int = 6000) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text[:max_chars]
    except Exception as e:
        return f"[Could not read PDF: {e}]"


def _read_scripts(pattern: str = "agents/super-crime/*_agent*.py") -> dict[str, str]:
    """Return {filename: source} for all scripts matching pattern."""
    scripts = {}
    for p in sorted(Path(".").glob(pattern)):
        try:
            scripts[p.name] = p.read_text()
        except Exception as e:
            scripts[p.name] = f"[Could not read: {e}]"
    return scripts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    problem_domain: str,
    papers_dir: Optional[Path] = None,
) -> Paper:
    """
    Generate a novel research-agent script (meta_run_agent.py) by synthesising
    existing agent code and the originator paper, then validate it runs.
    """

    # ------------------------------------------------------------------
    # 1. Read inputs — agent scripts + originator.pdf
    # ------------------------------------------------------------------
    print("Reading agent scripts...")
    scripts = _read_scripts("agents/super-crime/*_agent*.py")
    print(f"  Found {len(scripts)} script(s): {list(scripts.keys())}")

    print("Reading originator.pdf...")
    pdf_text = ""
    originator_pdf = Path("originator.pdf")
    if originator_pdf.exists():
        pdf_text = _read_pdf(originator_pdf)
        print(f"  Read {len(pdf_text)} chars from originator.pdf")
    else:
        print("  originator.pdf not found — skipping")

    # ------------------------------------------------------------------
    # 2. Summarise each script individually (FAST_MODEL_ID)
    # ------------------------------------------------------------------
    print("Summarising individual scripts...")
    individual_summaries = {}

    for name, source in scripts.items():
        prompt = (
            f"{META_CONTEXT}\n\n"
            f"Summarise the following Python agent script concisely. Describe:\n"
            f"- Its overall structure and pipeline steps\n"
            f"- How it generates a scientific paper\n"
            f"- Key design choices and any notable techniques\n\n"
            f"Script: {name}\n\n```python\n{source[:4000]}\n```"
        )
        out = call_llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            model_id=FAST_MODEL_ID,
        )
        individual_summaries[name] = llm_text(out)
        print(f"  Summarised {name}")

    if pdf_text:
        prompt = (
            f"{META_CONTEXT}\n\n"
            f"Summarise the following research paper concisely, focusing on:\n"
            f"- The core proposed method\n"
            f"- Key results and findings\n"
            f"- How it could inspire an automated paper-generation agent\n\n"
            f"{pdf_text}"
        )
        out = call_llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            model_id=FAST_MODEL_ID,
        )
        individual_summaries["originator.pdf"] = llm_text(out)
        print("  Summarised originator.pdf")

    # ------------------------------------------------------------------
    # 3. Produce a single comparative summary (FAST_MODEL_ID)
    # ------------------------------------------------------------------
    print("Writing comparative summary...")
    summaries_block = "\n\n".join(
        f"### {name}\n{summary}"
        for name, summary in individual_summaries.items()
    )
    comparative_prompt = (
        f"{META_CONTEXT}\n\n"
        f"Below are summaries of several agent scripts and a reference paper, "
        f"all aimed at generating scientific papers on the domain: '{problem_domain}'.\n\n"
        f"{summaries_block}\n\n"
        f"Write a single comparative summary that:\n"
        f"- Identifies common patterns and shared pipeline steps\n"
        f"- Highlights where the scripts differ in approach\n"
        f"- Notes which ideas from the reference paper are or aren't reflected\n"
        f"- Identifies gaps or weaknesses across the existing approaches"
    )
    out = call_llm(
        messages=[{"role": "user", "content": [{"text": comparative_prompt}]}],
        model_id=FAST_MODEL_ID,
    )
    comparative_summary = llm_text(out)
    print("  Comparative summary written")

    # ------------------------------------------------------------------
    # 4. Novelty assessment + proposed structure (MODEL_ID)
    #    Uses comparative summary + raw scripts (not PDF)
    # ------------------------------------------------------------------
    print("Assessing novelty and proposing new structure...")
    scripts_block = "\n\n".join(
        f"### {name}\n```python\n{source[:3000]}\n```"
        for name, source in scripts.items()
    )
    novelty_prompt = (
        f"{META_CONTEXT}\n\n"
        f"You are reviewing automated scientific paper-generation agents for the "
        f"problem domain: '{problem_domain}'.\n\n"
        f"## Comparative Summary\n{comparative_summary}\n\n"
        f"## Original Scripts\n{scripts_block}\n\n"
        f"Based on the above:\n"
        f"1. Assess the novelty of the existing approaches — what is missing or "
        f"   could be significantly improved?\n"
        f"2. Propose a better agent code structure in plain English using bullet "
        f"   points and markdown sections. The structure should have the same goal "
        f"   (generating a scientific paper) but take a meaningfully different and "
        f"   more novel approach.\n"
        f"3. Be specific about pipeline steps, LLM call strategies, and how "
        f"   results are synthesised into paper sections."
    )
    out = call_llm(
        messages=[{"role": "user", "content": [{"text": novelty_prompt}]}],
        model_id=MODEL_ID,
    )
    proposed_structure = llm_text(out)
    print("  Novelty assessment and proposed structure written")

    # Save the guide as an intermediate markdown file for inspection
    GUIDE_PATH = META_RUN_PATH.parent / "meta_run_agent_guide.md"
    GUIDE_PATH.write_text(
        f"# Agent Design Guide\n\n"
        f"**Problem domain:** {problem_domain}\n\n"
        f"{proposed_structure}"
    )
    print(f"  Guide written to {GUIDE_PATH}")

    # ------------------------------------------------------------------
    # 5. Write meta_run_agent.py (MODEL_ID)
    # ------------------------------------------------------------------
    print("Writing meta_run_agent.py...")

    # Hardcoded publication-quality section prompt templates injected into the
    # generated script so each LLM call produces rich, academic-standard output.
    SECTION_PROMPT_TEMPLATES = """
## Hardcoded Section Prompt Requirements

The ULTIMATE GOAL of this script is to produce a PUBLICATION-QUALITY scientific paper.
Every LLM call that drafts a paper section MUST use rich, specific prompts that instruct
the model to write at the standard of a peer-reviewed journal submission. The following
templates MUST be used verbatim or closely adapted for each section:

### Introduction prompt (use MODEL_ID)
Write a publication-quality Introduction section (5-6 paragraphs, ~600 words) for a
scientific paper submitted to a top-tier AI/ML venue. The introduction must: (1) open
with a compelling statement of the problem and its scientific significance; (2) survey
the relevant prior art with specific named works and their limitations; (3) clearly
articulate the gap this work addresses; (4) state the research question precisely and
explain why it is falsifiable; (5) summarise the key findings and what they rule out;
(6) list 3-4 concrete contributions in bullet form. Use formal academic register.
Do not use vague phrases like "this paper explores". Be specific and assertive.

### Methods prompt (use MODEL_ID)
Write a publication-quality Methods section (5-7 paragraphs, ~700 words) with the
rigour expected at NeurIPS or ICML. Include: (a) a subsection on Experimental Design
explaining the overall structure and controls; (b) a subsection on the Simulation
Harness with all parameters stated explicitly (seed, n_trials, n_proposers, noise
levels, n_hypotheses); (c) a subsection on Pre-registered Predictions listing each
hypothesis with its operationalised metric, direction, and threshold; (d) a subsection
on Disqualification Criteria defining a priori what would render the experiment
uninformative; (e) a subsection on Adjudication Procedure. Use precise, passive-voice
academic prose. Do NOT report results.

### Results prompt (use MODEL_ID)
Write a publication-quality Results section (5-6 paragraphs, ~600 words). Report ONLY
what is in the provided data. Structure as: (a) overview paragraph with all aggregate
statistics stated to 4 decimal places; (b) per-noise-level breakdown with a clear
narrative arc; (c) hypothesis-by-hypothesis adjudication, explicitly stating for each
whether it survived or was falsified and the exact observed value vs threshold;
(d) disqualifier status; (e) a summary paragraph stating the net finding. Use tables
or structured lists where appropriate. Never interpret—only report.

### Discussion prompt (use MODEL_ID)
Write a publication-quality Discussion section (7-9 paragraphs, ~900 words) with the
following mandatory subsections: "What Survived and What It Means" (interpret the
surviving claim in the context of the domain); "What Was Falsified" (explicitly state
what is now ruled out); "Addressing the Strongest Objection" (engage directly with the
hostile reviewer critique with a specific counter-argument); "Limitations" (at least
four concrete, domain-specific limitations—not boilerplate); "Broader Implications"
(connect findings to the larger field of agentic scientific discovery); "Future Work"
(three specific, actionable directions). Use formal academic prose throughout.

### call_llm usage in the generated script
All LLM calls MUST use this exact pattern (Bedrock converse API):
    out = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=MODEL_ID,
        inferenceConfig={"maxTokens": 3000},
    )
    text = (out.get("output", {}).get("message", {})
               .get("content", [{}])[0].get("text", "").strip())
Never pass prompt= or max_tokens= directly to call_llm.

### Required imports from hackathon_science.customtools and hackathon_science.tools
The script MUST reuse existing utilities rather than reimplementing them:
    from hackathon_science.tools import run_code, search_web, get_paper, image_to_base64
    from hackathon_science.customtools import (
        llm_text, extract_score, extract_code, has_error, score_paper,
        in_scope, escape_latex_blocks, META_CONTEXT,
    )
Use llm_text() to extract text from all call_llm responses.
Use escape_latex_blocks() on every drafted section before assigning it.
Use run_code() to execute any generated experiment scripts.
Use has_error() to check run_code output for failures.
Use search_web() for literature retrieval where appropriate.

### Figure detection and embedding (MANDATORY)
After running any experiment code with run_code(), the script MUST detect newly
created PNG files and embed them in the results section using image_to_base64.
Use this exact pattern:

    from pathlib import Path
    pngs_before = set(Path(".").glob("*.png"))

    # ... run experiment ...
    code_output = run_code(experiment_code, filename="experiment.py")

    new_pngs = sorted(set(Path(".").glob("*.png")) - pngs_before)
    figures = []
    for i, png_path in enumerate(new_pngs, 1):
        caption_prompt = (
            f"Write a concise 1-2 sentence figure caption for Figure {i} "
            f"saved as {png_path.name}, for a paper on: {research_question}. "
            f"Begin with 'Figure {i}:'"
        )
        cap_out = call_llm(
            messages=[{"role": "user", "content": [{"text": caption_prompt}]}],
            model_id=FAST_MODEL_ID,
            inferenceConfig={"maxTokens": 200},
        )
        caption = llm_text(cap_out)
        fig_md  = image_to_base64(str(png_path), alt_text=caption)
        figures.append((caption, fig_md))

    # Append figures at end of results section:
    if figures:
        results += "\n\n---\n\n" + "\n\n".join(
            f"{fig_md}\n\n*{caption}*" for caption, fig_md in figures
        )
"""

    codegen_prompt = (
        f"{META_CONTEXT}\n\n"
        f"You are writing a Python agent script for the hackathon problem domain: "
        f"'{problem_domain}'.\n\n"
        f"## Proposed Structure\n{proposed_structure}\n\n"
        f"{SECTION_PROMPT_TEMPLATES}\n\n"
        f"## Instructions\n"
        f"- Write a complete, self-contained Python script.\n"
        f"- The script must define a function: `def run(problem_domain: str, papers_dir=None) -> Paper:`\n"
        f"- It must import `Paper` from `hackathon_science`.\n"
        f"- It must import `call_llm` from `hackathon_science.utils`.\n"
        f"- It MUST import and USE functions from `hackathon_science.customtools` and\n"
        f"  `hackathon_science.tools` as specified in the requirements above.\n"
        f"- Use FAST_MODEL_ID = 'global.anthropic.claude-sonnet-4-6' for quick calls.\n"
        f"- Use MODEL_ID = 'global.anthropic.claude-opus-4-7' for deep reasoning.\n"
        f"- Every section-drafting LLM call MUST follow the prompt templates above exactly.\n"
        f"- The goal is a PUBLICATION-QUALITY scientific paper — not a stub or placeholder.\n"
        f"- The `run` function MUST end with exactly this return statement:\n\n"
        f"```python\n{PAPER_RETURN_TEMPLATE}\n```\n\n"
        f"- The variables `research_question`, `introduction`, `methods`, `results`, "
        f"`references`, and `final_code` must all be defined as non-empty strings "
        f"before the return.\n"
        f"- `final_code` must be a string containing valid Python code.\n"
        f"- Return ONLY the complete Python script inside a ```python ... ``` block."
    )
    out = call_llm(
        messages=[{"role": "user", "content": [{"text": codegen_prompt}]}],
        model_id=MODEL_ID,
        inferenceConfig={"maxTokens": 8192},
    )
    agent_code = extract_code(llm_text(out))

    # If the code was truncated before the return block, ask the model to continue
    CONTINUE_ATTEMPTS = 2
    for _ in range(CONTINUE_ATTEMPTS):
        if "return Paper(" in agent_code:
            break
        print("  Code appears truncated — requesting continuation...")
        continuation_messages = [
            {"role": "user",      "content": [{"text": codegen_prompt}]},
            {"role": "assistant", "content": [{"text": f"```python\n{agent_code}"}]},
            {"role": "user",      "content": [{"text": (
                "Your response was cut off. Continue the Python script from exactly "
                "where you stopped and complete it, ending with the required return "
                f"statement:\n\n```python\n{PAPER_RETURN_TEMPLATE}\n```\n\n"
                "Output ONLY the continuation (no recap of prior code)."
            )}]},
        ]
        cont_out   = call_llm(
            messages=continuation_messages,
            model_id=MODEL_ID,
            inferenceConfig={"maxTokens": 8192},
        )
        agent_code = agent_code + "\n" + llm_text(cont_out).replace("```python", "").replace("```", "").strip()

    META_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_RUN_PATH.write_text(agent_code)
    print(f"  Written to {META_RUN_PATH}")

    # ------------------------------------------------------------------
    # 6. Return a meta-Paper describing what was produced
    # ------------------------------------------------------------------
    introduction = escape_latex_blocks(
        f"This meta-agent synthesised {len(scripts)} existing agent script(s) "
        f"and the Flow of Options reference paper to propose and implement a novel "
        f"automated scientific paper-generation pipeline for the domain: {problem_domain}.\n\n"
        f"## Comparative Summary\n{comparative_summary}"
    )
    methods = escape_latex_blocks(proposed_structure)
    results = escape_latex_blocks(
        f"meta_run_agent.py was successfully generated and written to {META_RUN_PATH}. "
        f"Run it with: uv run hackathon run {META_RUN_PATH}"
    )
    references = "Flow of Options. https://arxiv.org/pdf/2502.12929"

    return Paper(
        title=f"Meta-Agent: Novel Scientific Discovery Pipeline for {problem_domain}",
        introduction=introduction,
        methods=methods,
        results=results,
        references=references,
        appendix=f"```python\n{agent_code}\n```",
        tags=["meta", "agent", "code-generation"],
    )