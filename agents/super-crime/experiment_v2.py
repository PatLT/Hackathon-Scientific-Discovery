"""
Improved experiment: 30 problems, harder items, 3 trials, statistical analysis.
Target: ~70% baseline accuracy to leave room for scaling effects.
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

from hackathon_science.utils import call_llm
from collections import Counter
import statistics

MODEL = "global.anthropic.claude-sonnet-4-6"

PROBLEMS = [
    # === ARITHMETIC (6 problems) - mix of easy and hard ===
    {"q": "What is 37 * 43?", "a": "1591", "cat": "arithmetic"},
    {"q": "What is 123 + 456 + 789?", "a": "1368", "cat": "arithmetic"},
    {"q": "What is 17 * 19?", "a": "323", "cat": "arithmetic"},
    {"q": "What is 847 - 389?", "a": "458", "cat": "arithmetic"},
    {"q": "What is 156 / 12?", "a": "13", "cat": "arithmetic"},
    {"q": "What is 23 * 17 + 9?", "a": "400", "cat": "arithmetic"},

    # === LOGIC (6 problems) - include tricky syllogisms ===
    {"q": "All bloops are razzies. All razzies are lazzies. Are all bloops lazzies? Yes or no.", "a": "yes", "cat": "logic"},
    {"q": "No fish are mammals. All dolphins are mammals. Are any dolphins fish? Yes or no.", "a": "no", "cat": "logic"},
    {"q": "All roses are flowers. Some flowers fade quickly. Must all roses fade quickly? Yes or no.", "a": "no", "cat": "logic"},
    {"q": "Some birds can fly. Penguins are birds. Can all penguins fly? Yes or no.", "a": "no", "cat": "logic"},
    {"q": "All A are B. All B are C. No C are D. Are any A also D? Yes or no.", "a": "no", "cat": "logic"},
    {"q": "If it rains, the ground is wet. The ground is wet. Did it rain? Yes or no.", "a": "no", "cat": "logic"},

    # === TRICK QUESTIONS (9 problems) - designed to trip up LLMs ===
    {"q": "How many times can you subtract 5 from 25?", "a": "1", "cat": "trick"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many are left?", "a": "9", "cat": "trick"},
    {"q": "If you have 3 apples and take away 2, how many do YOU have?", "a": "2", "cat": "trick"},
    {"q": "Some months have 30 days, some 31. How many have 28 days?", "a": "12", "cat": "trick"},
    {"q": "How many birthdays does the average person have?", "a": "1", "cat": "trick"},
    {"q": "A doctor gives you 3 pills, take one every 30 min. How many minutes until gone?", "a": "60", "cat": "trick"},
    {"q": "How many animals of each kind did Moses take on the ark?", "a": "0", "cat": "trick"},
    {"q": "Is it legal for a man to marry his widow's sister?", "a": "no", "cat": "trick"},
    {"q": "If there are 3 apples and you take 2, how many apples are there?", "a": "3", "cat": "trick"},

    # === PROBABILITY / BASE RATE (6 problems) - known LLM weaknesses ===
    {"q": "A test is 99% accurate. 1 in 1000 have the disease. You test positive. More likely sick or healthy?", "a": "healthy", "cat": "probability"},
    {"q": "I flip a fair coin 5 times, all heads. Probability next flip is heads? Answer as fraction.", "a": "1/2", "cat": "probability"},
    {"q": "In a room of 23 people, is it more likely than not that two share a birthday? Yes or no.", "a": "yes", "cat": "probability"},
    {"q": "Roll two dice. Is sum of 7 more or less likely than sum of 12?", "a": "more", "cat": "probability"},
    {"q": "I pick a random person. They have two kids, at least one is a boy. P(both boys)? Answer as fraction.", "a": "1/3", "cat": "probability"},
    {"q": "A bat and ball cost $1.10. Bat costs $1 more than ball. Ball costs how many cents?", "a": "5", "cat": "probability"},

    # === COUNTERFACTUAL (3 problems) ===
    {"q": "In a world where the sky is green, what color is the sky?", "a": "green", "cat": "counterfactual"},
    {"q": "If 2+2=5 in a hypothetical world, what is 2+2 in that world?", "a": "5", "cat": "counterfactual"},
    {"q": "In a world where dogs have 6 legs, how many legs does a dog have?", "a": "6", "cat": "counterfactual"},
]

N_TRIALS = 3


def get_answer(question):
    """Get single LLM answer with error handling."""
    try:
        resp = call_llm(
            messages=[{"role": "user", "content": [{"text": f"Answer with one word or number only: {question}"}]}],
            model_id=MODEL
        )
        text = resp.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        text = text.strip().lower().replace("*", "").replace("`", "").replace(".", "").replace(",", "")
        return text.split()[0] if text.split() else ""
    except Exception as e:
        return ""


def check_answer(got, expected):
    """Flexible answer matching."""
    got = got.lower().strip()
    expected = expected.lower().strip()

    if got == expected:
        return True

    # Number words
    nums = {
        "0": ["zero", "none", "no"], "1": ["one", "once"], "2": ["two"], "3": ["three"],
        "5": ["five"], "6": ["six"], "9": ["nine"], "12": ["twelve", "all"],
        "13": ["thirteen"], "60": ["sixty"], "1/2": ["half", "0.5", "50%"],
        "1/3": ["0.33", "33%", "one-third"]
    }
    for num, alts in nums.items():
        if expected == num and got in alts:
            return True
        if got == num and expected in alts:
            return True

    # Yes/no
    if expected == "yes" and got in ["yes", "true", "correct"]:
        return True
    if expected == "no" and got in ["no", "false", "incorrect", "not", "cannot", "dead"]:
        return True

    # More/less
    if expected == "more" and got in ["more", "higher", "greater", "likely"]:
        return True
    if expected == "healthy" and got in ["healthy", "not", "no", "unlikely"]:
        return True

    # Substring for longer answers
    if len(expected) > 2 and expected in got:
        return True

    return False


def majority_vote(question, n_agents):
    """Get majority vote from n agents."""
    votes = [get_answer(question) for _ in range(n_agents)]
    if not votes or all(v == "" for v in votes):
        return "", votes
    valid_votes = [v for v in votes if v]
    if not valid_votes:
        return "", votes
    winner = Counter(valid_votes).most_common(1)[0][0]
    return winner, votes


def run_trial(agent_counts):
    """Run one trial across all problems."""
    results = {n: {"correct": 0, "total": 0, "by_cat": {}} for n in agent_counts}

    for p in PROBLEMS:
        cat = p["cat"]
        for n in agent_counts:
            answer, _ = majority_vote(p["q"], n)
            correct = check_answer(answer, p["a"])

            results[n]["total"] += 1
            if correct:
                results[n]["correct"] += 1

            if cat not in results[n]["by_cat"]:
                results[n]["by_cat"][cat] = {"correct": 0, "total": 0}
            results[n]["by_cat"][cat]["total"] += 1
            if correct:
                results[n]["by_cat"][cat]["correct"] += 1

    return results


def main():
    agent_counts = [1, 3, 5, 7]
    all_trials = []

    print("=" * 70)
    print("FLOW-OF-OPTIONS SCALING EXPERIMENT")
    print(f"Model: {MODEL}")
    print(f"Problems: {len(PROBLEMS)} | Trials: {N_TRIALS} | Agent counts: {agent_counts}")
    print("=" * 70)

    for trial in range(N_TRIALS):
        print(f"\nTrial {trial + 1}/{N_TRIALS}...")
        results = run_trial(agent_counts)
        all_trials.append(results)
        for n in agent_counts:
            acc = results[n]["correct"] / results[n]["total"] * 100
            print(f"  n={n}: {acc:.1f}%")

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print("\n### Overall Accuracy (mean ± std across trials)")
    print("| Agents | Mean | Std | 95% CI | Raw |")
    print("|--------|------|-----|--------|-----|")

    for n in agent_counts:
        accs = [t[n]["correct"] / t[n]["total"] * 100 for t in all_trials]
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0
        ci = 1.96 * std / (len(accs) ** 0.5) if len(accs) > 1 else 0
        raw = ", ".join([f"{a:.0f}%" for a in accs])
        print(f"| {n} | {mean:.1f}% | {std:.1f}% | ±{ci:.1f}% | {raw} |")

    # Category breakdown (average across trials)
    print("\n### Accuracy by Category (mean across trials)")
    cats = sorted(set(p["cat"] for p in PROBLEMS))
    header = "| Category |" + "".join(f" n={n} |" for n in agent_counts)
    print(header)
    print("|----------|" + "------|" * len(agent_counts))

    for cat in cats:
        row = f"| {cat:8} |"
        for n in agent_counts:
            cat_accs = []
            for t in all_trials:
                c = t[n]["by_cat"].get(cat, {"correct": 0, "total": 1})
                cat_accs.append(c["correct"] / c["total"] * 100 if c["total"] > 0 else 0)
            mean_cat = statistics.mean(cat_accs)
            row += f" {mean_cat:4.0f}% |"
        print(row)

    # Scaling effect
    print("\n### Scaling Effect")
    baseline_accs = [t[1]["correct"] / t[1]["total"] * 100 for t in all_trials]
    max_accs = [t[7]["correct"] / t[7]["total"] * 100 for t in all_trials]
    baseline_mean = statistics.mean(baseline_accs)
    max_mean = statistics.mean(max_accs)
    delta = max_mean - baseline_mean

    print(f"Baseline (n=1): {baseline_mean:.1f}%")
    print(f"Maximum (n=7): {max_mean:.1f}%")
    print(f"Scaling effect: {delta:+.1f} percentage points")

    if abs(delta) < 2:
        print("Interpretation: NO SIGNIFICANT SCALING EFFECT")
    elif delta > 0:
        print("Interpretation: SCALING HELPS")
    else:
        print("Interpretation: SCALING HURTS")

    # Summary stats
    print(f"\n### Statistical Summary")
    print(f"Total problems: {len(PROBLEMS)}")
    print(f"Total trials: {N_TRIALS}")
    print(f"Total observations: {len(PROBLEMS) * len(agent_counts) * N_TRIALS}")
    print(f"LLM calls: ~{len(PROBLEMS) * sum(agent_counts) * N_TRIALS}")


if __name__ == "__main__":
    main()
