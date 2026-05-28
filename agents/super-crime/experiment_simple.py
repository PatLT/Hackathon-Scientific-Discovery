"""
Simple, reliable experiment: 20 problems, 1 trial, clean output.
Designed to run without errors.
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

from hackathon_science.utils import call_llm
from collections import Counter

MODEL = "global.anthropic.claude-sonnet-4-6"

PROBLEMS = [
    # Arithmetic (5)
    {"q": "What is 37 * 43?", "a": "1591", "cat": "arithmetic"},
    {"q": "What is 123 + 456 + 789?", "a": "1368", "cat": "arithmetic"},
    {"q": "What is 17 * 19?", "a": "323", "cat": "arithmetic"},
    {"q": "What is 144 / 12?", "a": "12", "cat": "arithmetic"},
    {"q": "What is 25 * 25?", "a": "625", "cat": "arithmetic"},

    # Logic (5)
    {"q": "All bloops are razzies. All razzies are lazzies. Are all bloops lazzies? Answer yes or no.", "a": "yes", "cat": "logic"},
    {"q": "No fish are mammals. All dolphins are mammals. Are any dolphins fish? Answer yes or no.", "a": "no", "cat": "logic"},
    {"q": "All roses are flowers. Some flowers fade quickly. Must all roses fade quickly? Answer yes or no.", "a": "no", "cat": "logic"},
    {"q": "Some artists are musicians. All musicians are creative. Are some artists creative? Answer yes or no.", "a": "yes", "cat": "logic"},
    {"q": "All squares are rectangles. All rectangles have four sides. Do all squares have four sides? Answer yes or no.", "a": "yes", "cat": "logic"},

    # Trick/CRT (5)
    {"q": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost in cents?", "a": "5", "cat": "trick"},
    {"q": "If 5 machines take 5 minutes to make 5 widgets, how many minutes for 100 machines to make 100 widgets?", "a": "5", "cat": "trick"},
    {"q": "A patch of lily pads doubles daily. It covers the lake on day 48. On what day was it half covered?", "a": "47", "cat": "trick"},
    {"q": "How many times can you subtract 5 from 25?", "a": "1", "cat": "trick"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many are left?", "a": "9", "cat": "trick"},

    # Semantic/Counterfactual (5)
    {"q": "In a world where the sky is green, what color is the sky?", "a": "green", "cat": "semantic"},
    {"q": "If you take 2 apples from 3 apples, how many apples do YOU have?", "a": "2", "cat": "semantic"},
    {"q": "Some months have 30 days, some 31. How many have 28 days?", "a": "12", "cat": "semantic"},
    {"q": "How many birthdays does the average person have?", "a": "1", "cat": "semantic"},
    {"q": "If 2+2=5 in a hypothetical world, what is 2+2 in that world?", "a": "5", "cat": "semantic"},
]


def get_answer(question):
    """Get single LLM answer."""
    try:
        resp = call_llm(
            messages=[{"role": "user", "content": [{"text": f"Answer with one word or number only: {question}"}]}],
            model_id=MODEL
        )
        text = resp.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        text = text.strip().lower().replace("*", "").replace("`", "").replace(".", "")
        return text.split()[0] if text.split() else ""
    except Exception as e:
        return f"ERROR:{e}"


def check_answer(got, expected):
    """Flexible answer matching."""
    got = got.lower().strip()
    expected = expected.lower().strip()

    if got == expected:
        return True

    # Number words
    nums = {"0": "zero", "1": "one", "2": "two", "5": "five", "9": "nine", "12": "twelve", "47": "forty-seven"}
    if expected in nums and got == nums[expected]:
        return True
    if expected in nums.values() and got in [k for k, v in nums.items() if v == expected]:
        return True

    # Yes/no
    if expected == "yes" and got in ["yes", "true", "correct"]:
        return True
    if expected == "no" and got in ["no", "false", "incorrect"]:
        return True

    # Substring
    if expected in got or got in expected:
        return True

    return False


def majority_vote(question, n_agents):
    """Get majority vote from n agents."""
    votes = [get_answer(question) for _ in range(n_agents)]
    winner = Counter(votes).most_common(1)[0][0]
    return winner, votes


def main():
    agent_counts = [1, 3, 5, 7]
    results = {n: {"correct": 0, "total": 0, "by_cat": {}} for n in agent_counts}

    print("=" * 60)
    print("FLOW-OF-OPTIONS SCALING EXPERIMENT")
    print(f"Model: {MODEL}")
    print(f"Problems: {len(PROBLEMS)} across {len(set(p['cat'] for p in PROBLEMS))} categories")
    print(f"Agent counts: {agent_counts}")
    print("=" * 60)

    for i, p in enumerate(PROBLEMS):
        print(f"\n[{i+1}/{len(PROBLEMS)}] {p['cat'].upper()}: {p['q'][:50]}...")
        print(f"  Expected: {p['a']}")

        for n in agent_counts:
            answer, votes = majority_vote(p["q"], n)
            correct = check_answer(answer, p["a"])

            results[n]["total"] += 1
            if correct:
                results[n]["correct"] += 1

            cat = p["cat"]
            if cat not in results[n]["by_cat"]:
                results[n]["by_cat"][cat] = {"correct": 0, "total": 0}
            results[n]["by_cat"][cat]["total"] += 1
            if correct:
                results[n]["by_cat"][cat]["correct"] += 1

            mark = "CORRECT" if correct else "WRONG"
            print(f"  n={n}: '{answer}' [{mark}]")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nOverall Accuracy:")
    print("| Agents | Correct | Total | Accuracy |")
    print("|--------|---------|-------|----------|")
    for n in agent_counts:
        r = results[n]
        acc = r["correct"] / r["total"] * 100
        print(f"| {n}      | {r['correct']}      | {r['total']}    | {acc:.1f}%     |")

    print("\nAccuracy by Category:")
    cats = sorted(set(p["cat"] for p in PROBLEMS))
    header = "| Category   |" + "".join(f" n={n} |" for n in agent_counts)
    print(header)
    print("|" + "-" * 11 + "|" + "------|" * len(agent_counts))
    for cat in cats:
        row = f"| {cat:10} |"
        for n in agent_counts:
            c = results[n]["by_cat"].get(cat, {"correct": 0, "total": 1})
            acc = c["correct"] / c["total"] * 100 if c["total"] > 0 else 0
            row += f" {acc:4.0f}% |"
        print(row)

    # Scaling effect
    print("\nScaling Effect:")
    base = results[1]["correct"] / results[1]["total"] * 100
    top = results[7]["correct"] / results[7]["total"] * 100
    delta = top - base
    print(f"  Baseline (n=1): {base:.1f}%")
    print(f"  Maximum (n=7):  {top:.1f}%")
    print(f"  Change: {delta:+.1f} percentage points")

    if delta > 2:
        print("  Interpretation: Scaling HELPS")
    elif delta < -2:
        print("  Interpretation: Scaling HURTS")
    else:
        print("  Interpretation: Scaling has NO EFFECT")


if __name__ == "__main__":
    main()
