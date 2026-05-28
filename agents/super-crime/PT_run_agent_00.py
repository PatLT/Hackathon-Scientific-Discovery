"""
Run agent template for super-crime.
Write your paper-generation logic here.
"""
from pathlib import Path
import re
import numpy as np
from typing import Optional
from hackathon_science import Paper
from hackathon_science.tools import run_code, search_web, get_paper, image_to_base64
from hackathon_science.utils import call_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_text(response: dict) -> str:
    """Safely extract text from a call_llm response dict."""
    output = response.get("output", {}).get("message", {}).get("content", [])
    return output[0].get("text", "").strip() if output else ""


def _extract_score(text: str) -> int:
    """Extract integer score from 'SCORE = NUMBER' pattern."""
    match = re.search(r'SCORE\s*=\s*(\d+(?:\.\d+)?)', text)
    return int(float(match.group(1))) if match else 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    problem_domain: str,
    papers_dir: Optional[Path] = None
) -> Paper:
    """
    Generate a research paper.

    Args:
        problem_domain: Research area prompt (same for all teams)
        papers_dir: Optional path to papers directory (use with get_paper)

    Returns:
        Paper with structured fields
    """

    FAST_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
    MODEL_ID      = "global.anthropic.claude-opus-4-7"

    # ------------------------------------------------------------------
    # 1. Search the web and score results for relevance
    # ------------------------------------------------------------------
    print("Searching web...")
    search_results = search_web(problem_domain, max_results=20)

    score_results = []
    for res in search_results:
        title   = res.get("title", "")
        snippet = res.get("snippet", "")
        prompt  = (
            f"Assess this title and snippet for relevance to topic '{problem_domain}'. "
            f"Return a number from 0-10 with 10 being highly relevant. "
            f"Return in format SCORE=NUMBER (no brackets). "
            f'Title: "{title}", Snippet: "{snippet}"'
        )
        out  = call_llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            model_id=FAST_MODEL_ID,
        )
        txt   = _llm_text(out)
        score = _extract_score(txt)
        score_results.append(score)
        print(f"  [{score:>2}] {title[:80]}")

    # ------------------------------------------------------------------
    # 2. Select top-12 most relevant results
    # ------------------------------------------------------------------
    n_top = min(12, len(search_results))
    top_indices = np.argsort(score_results)[-n_top:]  # ascending → take tail

    # ------------------------------------------------------------------
    # 3. Load any papers from papers_dir and append to search_results
    # ------------------------------------------------------------------
    if papers_dir:
        print("Loading papers from papers_dir...")
        papers_path = Path(papers_dir)
        paper_ids   = [p.stem for p in papers_path.iterdir() if p.is_file()]
        for paper_id in paper_ids:
            paper = get_paper(paper_id, papers_dir)
            if not paper:
                continue
            title   = paper.get("title",    "")
            snippet = paper.get("abstract", paper.get("introduction", ""))[:300]
            print(f"  Loaded: {title[:80]}")

            # Score for relevance just like web results
            prompt = (
                f"Assess this title and abstract for relevance to topic '{problem_domain}'. "
                f"Return a number from 0-10 with 10 being highly relevant. "
                f"Return in format SCORE=NUMBER (no brackets). "
                f'Title: "{title}", Abstract: "{snippet}"'
            )
            out   = call_llm(
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                model_id=FAST_MODEL_ID,
            )
            score = _extract_score(_llm_text(out))
            print(f"  [{score:>2}] {title[:80]}")

            # Append as a pseudo search-result entry so section 4 can treat
            # local and web papers uniformly
            search_results.append({
                "title":   title,
                "snippet": snippet,
                "url":     paper.get("url", f"local://{paper_id}"),
                "_paper":  paper,   # stash full paper for richer summarisation
            })
            score_results.append(score)

        # Re-select top indices now that the lists are longer
        n_top       = min(12, len(search_results))
        top_indices = np.argsort(score_results)[-n_top:]

    # ------------------------------------------------------------------
    # 4. Summarise each top paper and generate a research question from it
    # ------------------------------------------------------------------
    print("Summarising top papers and extracting questions...")
    intro_paragraphs = []
    questions        = [problem_domain + "."]  # seed the question list

    for cite_i in top_indices:
        res        = search_results[cite_i]
        url        = res.get("url", "")
        full_paper = res.get("_paper")  # present for local papers, None for web

        if full_paper:
            # Summarise directly from the stored text rather than fetching a URL
            paper_text = (
                full_paper.get("abstract", "")
                or full_paper.get("introduction", "")
                or full_paper.get("title", "")
            )
            summary_prompt = (
                f"Read the following paper excerpt and return a brief, concise "
                f"2-sentence summary:\n\n{paper_text[:1500]}"
            )
            question_prompt = (
                f"Read the following paper excerpt and formulate a single, concise "
                f"scientific question arising from it. Return ONLY the question as "
                f"a single sentence ending in a question mark:\n\n{paper_text[:1500]}"
            )
        else:
            summary_prompt = (
                f"Review this paper and return a brief, concise 2-sentence summary. "
                f"Link: {url}"
            )
            question_prompt = (
                f"Review this paper and formulate a single, concise, scientific question "
                f"arising from it. Return ONLY the question as a single sentence ending "
                f"in a question mark. Link: {url}"
            )

        out_summary = call_llm(
            messages=[{"role": "user", "content": [{"text": summary_prompt}]}],
            model_id=FAST_MODEL_ID,
        )
        summary = _llm_text(out_summary)
        intro_paragraphs.append(summary)

        out_question = call_llm(
            messages=[{"role": "user", "content": [{"text": question_prompt}]}],
            model_id=FAST_MODEL_ID,
        )
        question = _llm_text(out_question)
        questions.append(question)
        print(f"  Q: {question[:100]}")

    # ------------------------------------------------------------------
    # 5. Synthesise a single best research question — OUTSIDE the loop
    # ------------------------------------------------------------------
    print("Synthesising best research question...")
    q_list       = "\n".join(f"- {q}" for q in questions)
    synthesis_prompt = (
        f"Below is a list of scientific questions related to the domain "
        f"'{problem_domain}':\n\n{q_list}\n\n"
        f"Select or synthesise a single best scientific question that would "
        f"make a strong, novel research paper title. Return ONLY the question."
    )
    out_synthesis = call_llm(
        messages=[{"role": "user", "content": [{"text": synthesis_prompt}]}],
        model_id=MODEL_ID,
    )
    research_question = _llm_text(out_synthesis)
    print(f"Research question: {research_question}")

    # ------------------------------------------------------------------
    # 6. Write a methods section
    # ------------------------------------------------------------------
    print("Writing methods section...")
    methods_prompt = (
        f"Propose a rigorous method to answer the following research question:\n\n"
        f'"{research_question}"\n\n'
        f"The work relates to the problem domain: {problem_domain}.\n"
        f"Write the full output as a Methods section of a scientific paper, "
        f"including subsections for data, approach, and evaluation."
    )
    out_methods = call_llm(
        messages=[{"role": "user", "content": [{"text": methods_prompt}]}],
        model_id=MODEL_ID,
    )
    methods = _llm_text(out_methods)

    # ------------------------------------------------------------------
    # 7. Write an experiment / analysis script and run it (with retry)
    # ------------------------------------------------------------------
    _ERROR_SIGNALS = ("error", "traceback", "exception", "exit code 1", "syntaxerror",
                      "indentationerror", "nameerror", "typeerror", "valueerror")

    def _has_error(output: str) -> bool:
        return any(sig in output.lower() for sig in _ERROR_SIGNALS)

    print("Generating experiment code...")
    code_prompt = (
        f"Write a self-contained Python script that implements the methodology "
        f"described below. The script should print key results and save any "
        f"plots as PNG files in the current directory.\n\n"
        f"Methodology:\n{methods}"
    )
    messages = [{"role": "user", "content": [{"text": code_prompt}]}]
    out_code  = call_llm(messages=messages, model_id=MODEL_ID)
    code_text = _llm_text(out_code)

    # run_code accepts raw LLM responses (including markdown fences) directly.
    # If execution fails, feed the error back to the LLM and retry up to 3 times.
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Running experiment (attempt {attempt}/{MAX_RETRIES})...")
        code_output = run_code(code_text, filename="experiment.py")
        print(f"  Output preview: {code_output[:300]}")

        if not _has_error(code_output):
            print("  Experiment ran successfully.")
            break

        if attempt == MAX_RETRIES:
            print("  Max retries reached — proceeding with error output.")
            break

        print(f"  Error detected, asking LLM to fix...")
        # Extend the conversation so the LLM has full context
        messages = messages + [
            {"role": "assistant", "content": [{"text": code_text}]},
            {"role": "user",      "content": [{"text": (
                f"The script failed with the following error:\n\n{code_output}\n\n"
                f"Please fix the code and return a corrected, complete Python script."
            )}]},
        ]
        out_code  = call_llm(messages=messages, model_id=MODEL_ID)
        code_text = _llm_text(out_code)

    # ------------------------------------------------------------------
    # 8. Interpret results
    # ------------------------------------------------------------------
    print("Interpreting results...")
    results_prompt = (
        f"The following experiment was run to address the research question:\n"
        f'"{research_question}"\n\n'
        f"Experiment output:\n{code_output}\n\n"
        f"Write a Results section for a scientific paper interpreting these "
        f"findings clearly and concisely."
    )
    out_results = call_llm(
        messages=[{"role": "user", "content": [{"text": results_prompt}]}],
        model_id=MODEL_ID,
    )
    results = _llm_text(out_results)

    # ------------------------------------------------------------------
    # 9. Write introduction from collected summaries
    # ------------------------------------------------------------------
    intro_body    = "\n\n".join(intro_paragraphs)
    intro_prompt  = (
        f"Using the following background summaries, write a coherent Introduction "
        f"section for a scientific paper addressing the question:\n"
        f'"{research_question}"\n\n'
        f"Background summaries:\n{intro_body}"
    )
    out_intro = call_llm(
        messages=[{"role": "user", "content": [{"text": intro_prompt}]}],
        model_id=MODEL_ID,
    )
    introduction = _llm_text(out_intro)

    # ------------------------------------------------------------------
    # 10. Build references list
    # ------------------------------------------------------------------
    references_lines = []
    for i, idx in enumerate(top_indices, 1):
        res     = search_results[idx]
        title_r = res.get("title", "Unknown")
        url_r   = res.get("url",   "")
        references_lines.append(f"[{i}] {title_r}. {url_r}")
    references = "\n".join(references_lines)

    # ------------------------------------------------------------------
    # 11. Return the completed Paper
    # ------------------------------------------------------------------
    return Paper(
        title=research_question,
        introduction=introduction,
        methods=methods,
        results=results,
        references=references,
        appendix=code_text,
        tags=[],
    )