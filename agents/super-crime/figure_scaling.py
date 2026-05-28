"""
Generate panel figure: (A) overall accuracy vs agent count, (B) accuracy by problem type.
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

import matplotlib.pyplot as plt
from collections import Counter
from hackathon_science.utils import call_llm

MODEL = "global.anthropic.claude-sonnet-4-6"

PROBLEMS = [
    {"q": "What is 37 * 43?", "a": "1591", "type": "math"},
    {"q": "What is 123 + 456 + 789?", "a": "1368", "type": "math"},
    {"q": "If you have 3 apples and take away 2, how many do YOU have?", "a": "2", "type": "math"},
    {"q": "No wugs are tims. Some tims are pobs. Can we conclude that some wugs are not pobs? Answer yes or no.", "a": "no", "type": "logic"},
    {"q": "All cats are animals. Some animals are not pets. Can we conclude some cats are not pets? Answer yes or no.", "a": "no", "type": "logic"},
    {"q": "In a world where the sky is green, what color is the sky?", "a": "green", "type": "counterfactual"},
    {"q": "How many times can you subtract 5 from 25?", "a": "1", "type": "trick"},
    {"q": "A clerk at a butcher shop stands five feet ten inches tall and wears size 13 sneakers. What does he weigh?", "a": "meat", "type": "trick"},
    {"q": "Moses took how many of each animal on the ark?", "a": "0", "type": "trick"},
    {"q": "Some months have 30 days, some have 31. How many months have 28 days?", "a": "12", "type": "trick"},
]

AGENT_COUNTS = [1, 3, 5, 7]


def get_answer(problem: str) -> str:
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": f"Answer with just one word or number, no explanation: {problem}"}]}],
        model_id=MODEL
    )
    text = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip().lower()
    text = text.replace("*", "").replace("`", "").replace(".", "").replace(",", "")
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
    """Run experiment and return structured results."""
    results = {n: [] for n in AGENT_COUNTS}
    by_type = {t: {n: [] for n in AGENT_COUNTS} for t in set(p["type"] for p in PROBLEMS)}

    print("Running experiment...")
    for i, p in enumerate(PROBLEMS):
        print(f"  Problem {i+1}/{len(PROBLEMS)}: {p['q'][:40]}...")
        for n in AGENT_COUNTS:
            votes = [get_answer(p["q"]) for _ in range(n)]
            winner = Counter(votes).most_common(1)[0][0]
            correct = check(winner, p["a"])
            results[n].append(correct)
            by_type[p["type"]][n].append(correct)

    return results, by_type


def generate_figure(results: dict, by_type: dict, output_path: str = "figure_scaling.png"):
    """Generate 2-panel figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Overall accuracy bar chart
    accuracies = [sum(results[n]) / len(results[n]) * 100 for n in AGENT_COUNTS]
    bars = ax1.bar(AGENT_COUNTS, accuracies, color='steelblue', edgecolor='black', width=1.5)
    ax1.set_xlabel('Number of Agents', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('(A) Overall Accuracy vs. Agent Count', fontsize=12, fontweight='bold')
    ax1.set_xticks(AGENT_COUNTS)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='chance')
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{acc:.0f}%',
                ha='center', va='bottom', fontsize=10)

    # Panel B: Accuracy by problem type
    colors = {'math': '#2ecc71', 'logic': '#e74c3c', 'trick': '#9b59b6', 'counterfactual': '#f39c12'}
    markers = {'math': 'o', 'logic': 's', 'trick': '^', 'counterfactual': 'D'}

    for ptype, type_results in by_type.items():
        if not type_results[AGENT_COUNTS[0]]:
            continue
        accs = [sum(type_results[n]) / len(type_results[n]) * 100 for n in AGENT_COUNTS]
        ax2.plot(AGENT_COUNTS, accs, marker=markers.get(ptype, 'o'),
                label=ptype, color=colors.get(ptype, 'gray'), linewidth=2, markersize=8)

    ax2.set_xlabel('Number of Agents', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('(B) Accuracy by Problem Type', fontsize=12, fontweight='bold')
    ax2.set_xticks(AGENT_COUNTS)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {output_path}")
    return output_path


if __name__ == "__main__":
    results, by_type = run_experiment()
    generate_figure(results, by_type)
