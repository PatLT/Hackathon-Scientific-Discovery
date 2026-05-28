"""
Harder ablation: problems where models actually disagree/fail
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

from hackathon_science.utils import call_llm
from collections import Counter

# Expanded benchmark: 30 problems across 6 categories for stronger evidence
PROBLEMS = [
    # === CATEGORY 1: ARITHMETIC (5 problems) ===
    {"q": "What is 37 * 43?", "a": "1591", "cat": "arithmetic"},
    {"q": "What is 123 + 456 + 789?", "a": "1368", "cat": "arithmetic"},
    {"q": "What is 17 * 19?", "a": "323", "cat": "arithmetic"},
    {"q": "What is 1000 - 7 - 7 - 7 - 7 - 7?", "a": "965", "cat": "arithmetic"},
    {"q": "What is 144 / 12?", "a": "12", "cat": "arithmetic"},

    # === CATEGORY 2: SYLLOGISTIC LOGIC (5 problems) ===
    {"q": "No wugs are tims. Some tims are pobs. Can we conclude that some wugs are not pobs? Answer yes or no.", "a": "no", "cat": "logic"},
    {"q": "All cats are animals. Some animals are not pets. Can we conclude some cats are not pets? Answer yes or no.", "a": "no", "cat": "logic"},
    {"q": "All roses are flowers. All flowers need water. Do all roses need water? Answer yes or no.", "a": "yes", "cat": "logic"},
    {"q": "No fish are birds. All sparrows are birds. Are any sparrows fish? Answer yes or no.", "a": "no", "cat": "logic"},
    {"q": "Some doctors are tall. All tall people can reach high shelves. Can some doctors reach high shelves? Answer yes or no.", "a": "yes", "cat": "logic"},

    # === CATEGORY 3: TRICK QUESTIONS / CRT-style (5 problems) ===
    {"q": "How many times can you subtract 5 from 25?", "a": "1", "cat": "trick"},
    {"q": "A clerk at a butcher shop stands five feet ten inches tall and wears size 13 sneakers. What does he weigh?", "a": "meat", "cat": "trick"},
    {"q": "A bat and ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost in cents?", "a": "5", "cat": "trick"},
    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?", "a": "5", "cat": "trick"},
    {"q": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake?", "a": "47", "cat": "trick"},

    # === CATEGORY 4: MOSES ILLUSION / SEMANTIC TRAPS (5 problems) ===
    {"q": "Moses took how many of each animal on the ark?", "a": "0", "cat": "semantic"},
    {"q": "If you have 3 apples and take away 2, how many do YOU have?", "a": "2", "cat": "semantic"},
    {"q": "Some months have 30 days, some have 31. How many months have 28 days?", "a": "12", "cat": "semantic"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many are left?", "a": "9", "cat": "semantic"},
    {"q": "How many birthdays does the average person have?", "a": "1", "cat": "semantic"},

    # === CATEGORY 5: COUNTERFACTUAL REASONING (5 problems) ===
    {"q": "In a world where the sky is green, what color is the sky?", "a": "green", "cat": "counterfactual"},
    {"q": "Imagine dogs could fly. Could a flying dog catch a bird? Answer yes or no.", "a": "yes", "cat": "counterfactual"},
    {"q": "If water flowed uphill, would rivers flow from the ocean to mountains? Answer yes or no.", "a": "yes", "cat": "counterfactual"},
    {"q": "In a world where 2+2=5, what is 2+2?", "a": "5", "cat": "counterfactual"},
    {"q": "If humans had 4 arms, how many arms would a human have?", "a": "4", "cat": "counterfactual"},

    # === CATEGORY 6: BASE RATE / PROBABILITY (5 problems) ===
    {"q": "A test is 99% accurate. 1 in 1000 people have the disease. If someone tests positive, are they more likely to have the disease or not? Answer 'have' or 'not'.", "a": "not", "cat": "probability"},
    {"q": "You flip a fair coin 5 times and get heads each time. What is the probability the next flip is heads? Answer as a fraction.", "a": "1/2", "cat": "probability"},
    {"q": "In a room of 23 people, is it more likely than not that two share a birthday? Answer yes or no.", "a": "yes", "cat": "probability"},
    {"q": "If you roll two dice, is getting a sum of 7 more or less likely than getting a sum of 12? Answer 'more' or 'less'.", "a": "more", "cat": "probability"},
    {"q": "A family has two children. At least one is a boy. What is the probability both are boys? Answer as a fraction.", "a": "1/3", "cat": "probability"},
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
    if expected == "yes" and answer in ["yes", "true", "correct"]:
        return True
    if expected == "no" and answer in ["no", "false", "cannot", "incorrect"]:
        return True
    if expected.isdigit() and answer.isdigit() and answer == expected:
        return True
    # Handle numeric equivalents
    if expected == "0" and answer in ["0", "zero", "none"]:
        return True
    if expected == "1" and answer in ["1", "one"]:
        return True
    if expected == "2" and answer in ["2", "two"]:
        return True
    # Handle fractions
    if expected == "1/2" and answer in ["1/2", "0.5", "50%", "half"]:
        return True
    if expected == "1/3" and answer in ["1/3", "0.33", "33%"]:
        return True
    # Handle probability questions
    if expected == "have" and answer in ["have", "has", "likely"]:
        return True
    if expected == "not" and answer in ["not", "don't", "unlikely"]:
        return True
    if expected == "more" and answer in ["more", "higher", "greater"]:
        return True
    if expected == "less" and answer in ["less", "lower", "fewer"]:
        return True
    if expected in answer:
        return True
    return False


def run_experiment():
    agent_counts = [1, 3, 5, 7]
    results = {n: [] for n in agent_counts}
    category_results = {n: {} for n in agent_counts}
    details = []

    print("=" * 60)
    print("FLOW-OF-OPTIONS SCALING: EXPANDED BENCHMARK (30 PROBLEMS)")
    print("=" * 60)
    print(f"Categories: arithmetic, logic, trick, semantic, counterfactual, probability")
    print(f"Total problems: {len(PROBLEMS)}")

    for p in PROBLEMS:
        cat = p.get("cat", "unknown")
        print(f"\n[{cat}] Q: {p['q'][:50]}...")
        print(f"Expected: {p['a']}")
        row = {"question": p["q"][:40], "expected": p["a"], "category": cat}

        for n in agent_counts:
            votes = [get_answer(p["q"]) for _ in range(n)]
            winner = Counter(votes).most_common(1)[0][0]
            correct = check(winner, p["a"])
            results[n].append(correct)

            if cat not in category_results[n]:
                category_results[n][cat] = []
            category_results[n][cat].append(correct)

            status = '✓' if correct else '✗'
            print(f"  n={n}: {winner} (votes: {votes[:3]}{'...' if len(votes) > 3 else ''}) {status}")
            row[f"n{n}"] = "✓" if correct else "✗"

        details.append(row)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS SUMMARY")
    print("=" * 60)
    print("\n| Agents | Correct | Total | Accuracy |")
    print("|--------|---------|-------|----------|")
    for n in agent_counts:
        correct = sum(results[n])
        total = len(results[n])
        acc = correct / total
        print(f"| {n} | {correct} | {total} | {acc:.1%} |")

    # Per-category breakdown
    print("\n" + "=" * 60)
    print("PER-CATEGORY ACCURACY")
    print("=" * 60)
    categories = sorted(set(p.get("cat", "unknown") for p in PROBLEMS))
    header = "| Category |" + "".join(f" n={n} |" for n in agent_counts)
    print(header)
    print("|" + "-" * 14 + "|" + "------|" * len(agent_counts))
    for cat in categories:
        row = f"| {cat:<12} |"
        for n in agent_counts:
            if cat in category_results[n]:
                acc = sum(category_results[n][cat]) / len(category_results[n][cat])
                row += f" {acc:.0%}  |"
            else:
                row += " -    |"
        print(row)

    # Show per-problem breakdown
    print("\n" + "=" * 60)
    print("PER-PROBLEM BREAKDOWN")
    print("=" * 60)
    print("| Category | Problem | Expected | n=1 | n=3 | n=5 | n=7 |")
    print("|----------|---------|----------|-----|-----|-----|-----|")
    for row in details:
        print(f"| {row['category'][:8]} | {row['question'][:25]}... | {row['expected'][:6]} | {row.get('n1','-')} | {row.get('n3','-')} | {row.get('n5','-')} | {row.get('n7','-')} |")

    # Statistical summary
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"Total trials: {len(PROBLEMS) * len(agent_counts) * max(agent_counts)} LLM calls")
    print(f"Problems per category: {len(PROBLEMS) // len(categories)}")
    for n in agent_counts:
        correct = sum(results[n])
        total = len(results[n])
        print(f"n={n}: {correct}/{total} correct ({correct/total:.1%})")


if __name__ == "__main__":
    run_experiment()
