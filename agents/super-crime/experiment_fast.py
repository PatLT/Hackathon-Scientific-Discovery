"""
Quick ablation: agent count vs accuracy (fewer problems for speed)
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

from hackathon_science.utils import call_llm
from collections import Counter

PROBLEMS = [
    {"q": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost in cents?", "a": "5"},
    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?", "a": "5"},
    {"q": "In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half the lake?", "a": "47"},
    {"q": "All bloops are razzies. All razzies are lazzies. Are all bloops definitely lazzies? Answer yes or no.", "a": "yes"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "a": "9"},
]

MODEL = "global.anthropic.claude-sonnet-4-6"


def get_answer(problem: str) -> str:
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": f"Answer with just one word or number: {problem}"}]}],
        model_id=MODEL
    )
    text = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip().lower()
    text = text.replace("*", "").replace("`", "")
    for word in text.split():
        word = word.strip(".,!?")
        if word in ["yes", "no"] or word.isdigit():
            return word
    return text.split()[0] if text else ""


def check(answer: str, expected: str) -> bool:
    answer = answer.lower().strip()
    if answer == expected:
        return True
    if expected == "yes" and answer in ["yes", "true"]:
        return True
    if expected.isdigit() and expected in answer:
        return True
    return False


def run_experiment():
    agent_counts = [1, 3, 5]
    results = {n: [] for n in agent_counts}

    print("=" * 60)
    print("FLOW-OF-OPTIONS SCALING STUDY")
    print("=" * 60)

    for p in PROBLEMS:
        print(f"\nQ: {p['q'][:50]}...")
        for n in agent_counts:
            votes = [get_answer(p["q"]) for _ in range(n)]
            winner = Counter(votes).most_common(1)[0][0]
            correct = check(winner, p["a"])
            results[n].append(correct)
            print(f"  n={n}: {winner} {'✓' if correct else '✗'}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("| Agents | Accuracy |")
    print("|--------|----------|")
    for n in agent_counts:
        acc = sum(results[n]) / len(results[n])
        print(f"| {n} | {acc:.0%} ({sum(results[n])}/{len(results[n])}) |")


if __name__ == "__main__":
    run_experiment()
