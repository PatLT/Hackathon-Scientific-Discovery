"""
Harder ablation: problems where models actually disagree/fail
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

from hackathon_science.utils import call_llm
from collections import Counter

# Problems designed to trip up LLMs - ambiguous, tricky, or require careful reasoning
PROBLEMS = [
    # Tricky math
    {"q": "What is 37 * 43?", "a": "1591"},
    {"q": "What is 123 + 456 + 789?", "a": "1368"},
    {"q": "If you have 3 apples and take away 2, how many do YOU have?", "a": "2"},

    # Logic with negation (hard for LLMs)
    {"q": "No wugs are tims. Some tims are pobs. Can we conclude that some wugs are not pobs? Answer yes or no.", "a": "no"},
    {"q": "All cats are animals. Some animals are not pets. Can we conclude some cats are not pets? Answer yes or no.", "a": "no"},

    # Counterfactual reasoning
    {"q": "In a world where the sky is green, what color is the sky?", "a": "green"},

    # Trick questions
    {"q": "How many times can you subtract 5 from 25?", "a": "1"},
    {"q": "A clerk at a butcher shop stands five feet ten inches tall and wears size 13 sneakers. What does he weigh?", "a": "meat"},

    # Requires careful reading
    {"q": "Moses took how many of each animal on the ark?", "a": "0"},
    {"q": "Some months have 30 days, some have 31. How many months have 28 days?", "a": "12"},
]

MODEL = "global.anthropic.claude-sonnet-4-6"


def get_answer(problem: str) -> str:
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": f"Answer with just one word or number, no explanation: {problem}"}]}],
        model_id=MODEL
    )
    text = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip().lower()
    text = text.replace("*", "").replace("`", "").replace(".", "").replace(",", "")
    # Return first word/number
    for word in text.split():
        word = word.strip(".,!?\"'")
        if word:
            return word
    return text


def check(answer: str, expected: str) -> bool:
    answer = answer.lower().strip()
    expected = expected.lower().strip()
    if answer == expected:
        return True
    if expected == "yes" and answer in ["yes", "true"]:
        return True
    if expected == "no" and answer in ["no", "false", "cannot"]:
        return True
    if expected.isdigit() and answer.isdigit() and answer == expected:
        return True
    if expected in answer:
        return True
    return False


def run_experiment():
    agent_counts = [1, 3, 5, 7]
    results = {n: [] for n in agent_counts}
    details = []

    print("=" * 60)
    print("FLOW-OF-OPTIONS SCALING: HARD REASONING PROBLEMS")
    print("=" * 60)

    for p in PROBLEMS:
        print(f"\nQ: {p['q'][:55]}...")
        print(f"Expected: {p['a']}")
        row = {"question": p["q"][:40], "expected": p["a"]}

        for n in agent_counts:
            votes = [get_answer(p["q"]) for _ in range(n)]
            winner = Counter(votes).most_common(1)[0][0]
            correct = check(winner, p["a"])
            results[n].append(correct)
            status = '✓' if correct else '✗'
            print(f"  n={n}: {winner} (votes: {votes[:3]}{'...' if len(votes) > 3 else ''}) {status}")
            row[f"n{n}"] = "✓" if correct else "✗"

        details.append(row)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\n| Agents | Correct | Total | Accuracy |")
    print("|--------|---------|-------|----------|")
    for n in agent_counts:
        correct = sum(results[n])
        total = len(results[n])
        acc = correct / total
        print(f"| {n} | {correct} | {total} | {acc:.0%} |")

    # Show per-problem breakdown
    print("\n" + "=" * 60)
    print("PER-PROBLEM BREAKDOWN")
    print("=" * 60)
    print("| Problem | Expected | n=1 | n=3 | n=5 | n=7 |")
    print("|---------|----------|-----|-----|-----|-----|")
    for row in details:
        print(f"| {row['question'][:30]}... | {row['expected']} | {row.get('n1','-')} | {row.get('n3','-')} | {row.get('n5','-')} | {row.get('n7','-')} |")


if __name__ == "__main__":
    run_experiment()
