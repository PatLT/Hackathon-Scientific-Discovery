"""
Robust ablation study: 30+ problems, semantic matching, multiple trials with error bars.
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

from hackathon_science.utils import call_llm
from collections import Counter
import statistics

MODEL = "global.anthropic.claude-sonnet-4-6"

# Expanded problem set: 30 problems across categories
PROBLEMS = [
    # === COGNITIVE REFLECTION TEST (10 items) ===
    {"q": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost in cents?", "a": ["5", "five", "5 cents"], "category": "CRT"},
    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?", "a": ["5", "five"], "category": "CRT"},
    {"q": "In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half the lake?", "a": ["47", "forty-seven", "forty seven"], "category": "CRT"},
    {"q": "How many times can you subtract 5 from 25?", "a": ["1", "one", "once"], "category": "CRT"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "a": ["9", "nine"], "category": "CRT"},
    {"q": "If you have 3 apples and take away 2, how many apples do YOU have?", "a": ["2", "two"], "category": "CRT"},
    {"q": "Some months have 30 days, some have 31. How many months have 28 days?", "a": ["12", "twelve", "all"], "category": "CRT"},
    {"q": "A clerk at a butcher shop stands five feet ten inches tall and wears size 13 sneakers. What does he weigh?", "a": ["meat", "he weighs meat"], "category": "CRT"},
    {"q": "How many birthdays does the average person have?", "a": ["1", "one"], "category": "CRT"},
    {"q": "If a doctor gives you 3 pills and tells you to take one every half hour, how many minutes will the pills last?", "a": ["60", "sixty", "one hour", "1 hour"], "category": "CRT"},

    # === SYLLOGISTIC REASONING (10 items) ===
    {"q": "All roses are flowers. Some flowers fade quickly. Can we conclude all roses fade quickly? Answer yes or no.", "a": ["no"], "category": "logic"},
    {"q": "All bloops are razzies. All razzies are lazzies. Are all bloops definitely lazzies? Answer yes or no.", "a": ["yes"], "category": "logic"},
    {"q": "Some gribs are zorps. All zorps are flims. Are some gribs definitely flims? Answer yes or no.", "a": ["yes"], "category": "logic"},
    {"q": "No wugs are tims. Some tims are pobs. Can we conclude that some wugs are not pobs? Answer yes or no.", "a": ["no"], "category": "logic"},
    {"q": "All mammals are warm-blooded. All whales are mammals. Are all whales warm-blooded? Answer yes or no.", "a": ["yes"], "category": "logic"},
    {"q": "All cats are animals. Some animals are not pets. Can we conclude some cats are not pets? Answer yes or no.", "a": ["no"], "category": "logic"},
    {"q": "No fish are mammals. All dolphins are mammals. Are any dolphins fish? Answer yes or no.", "a": ["no"], "category": "logic"},
    {"q": "Some artists are musicians. All musicians are creative. Are some artists definitely creative? Answer yes or no.", "a": ["yes"], "category": "logic"},
    {"q": "All squares are rectangles. All rectangles have four sides. Do all squares have four sides? Answer yes or no.", "a": ["yes"], "category": "logic"},
    {"q": "Some books are paperbacks. No paperbacks are hardcovers. Are some books definitely not hardcovers? Answer yes or no.", "a": ["yes"], "category": "logic"},

    # === ARITHMETIC (10 items) ===
    {"q": "What is 37 * 43?", "a": ["1591"], "category": "math"},
    {"q": "What is 123 + 456 + 789?", "a": ["1368"], "category": "math"},
    {"q": "What is 17 * 24?", "a": ["408"], "category": "math"},
    {"q": "What is 999 - 567?", "a": ["432"], "category": "math"},
    {"q": "What is 144 / 12?", "a": ["12", "twelve"], "category": "math"},
    {"q": "What is 25 * 25?", "a": ["625"], "category": "math"},
    {"q": "What is 1000 - 777?", "a": ["223"], "category": "math"},
    {"q": "What is 15 * 15?", "a": ["225"], "category": "math"},
    {"q": "What is 81 / 9?", "a": ["9", "nine"], "category": "math"},
    {"q": "What is 19 + 28 + 37?", "a": ["84"], "category": "math"},
]

# Number of trials per condition for statistical robustness
N_TRIALS = 3


def get_answer(problem: str) -> str:
    """Get one agent's answer."""
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": f"Answer with just one word or number, no explanation: {problem}"}]}],
        model_id=MODEL
    )
    text = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip().lower()
    # Clean up formatting
    text = text.replace("*", "").replace("`", "").replace(".", "").replace(",", "")
    # Return first meaningful token
    for word in text.split():
        word = word.strip(".,!?\"'")
        if word:
            return word
    return text


def semantic_match(answer: str, expected_list: list[str]) -> bool:
    """Check if answer semantically matches any expected value."""
    answer = answer.lower().strip().replace(".", "").replace(",", "")

    for expected in expected_list:
        expected = expected.lower().strip()

        # Direct match
        if answer == expected:
            return True

        # Number normalization: "5" matches "five", etc.
        number_words = {
            "0": ["zero", "none"], "1": ["one", "once"], "2": ["two"], "3": ["three"],
            "4": ["four"], "5": ["five"], "6": ["six"], "7": ["seven"], "8": ["eight"],
            "9": ["nine"], "10": ["ten"], "11": ["eleven"], "12": ["twelve"],
            "47": ["forty-seven", "forty seven", "fortyseven"],
            "60": ["sixty"], "84": ["eighty-four", "eighty four"],
        }

        for num, words in number_words.items():
            if expected == num and answer in words:
                return True
            if answer == num and expected in words:
                return True

        # Yes/no normalization
        if expected in ["yes", "true", "valid", "correct"] and answer in ["yes", "true", "valid", "correct"]:
            return True
        if expected in ["no", "false", "invalid", "incorrect", "cannot"] and answer in ["no", "false", "invalid", "incorrect", "cannot"]:
            return True

        # Substring match for longer answers
        if len(expected) > 3 and expected in answer:
            return True
        if len(answer) > 3 and answer in expected:
            return True

    return False


def flow_of_options(problem: str, n_agents: int) -> tuple[str, list[str]]:
    """Run Flow-of-Options: n agents propose, then majority vote."""
    votes = [get_answer(problem) for _ in range(n_agents)]
    winner = Counter(votes).most_common(1)[0][0]
    return winner, votes


def run_single_trial(agent_counts: list[int]) -> dict:
    """Run one complete trial across all problems and agent counts."""
    results = {n: [] for n in agent_counts}
    category_results = {n: {"CRT": [], "logic": [], "math": []} for n in agent_counts}

    for p in PROBLEMS:
        for n in agent_counts:
            answer, _ = flow_of_options(p["q"], n)
            correct = semantic_match(answer, p["a"])
            results[n].append(correct)
            category_results[n][p["category"]].append(correct)

    return results, category_results


def run_experiment():
    """Run multiple trials and compute statistics."""
    agent_counts = [1, 3, 5, 7]

    print("=" * 70)
    print("FLOW-OF-OPTIONS SCALING: ROBUST EVALUATION")
    print(f"Problems: {len(PROBLEMS)} | Trials: {N_TRIALS} | Agent counts: {agent_counts}")
    print("=" * 70)

    # Collect results across trials
    all_trial_results = []
    all_category_results = []

    for trial in range(N_TRIALS):
        print(f"\nRunning trial {trial + 1}/{N_TRIALS}...")
        results, category_results = run_single_trial(agent_counts)
        all_trial_results.append(results)
        all_category_results.append(category_results)

    # Compute statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean ± std across trials)")
    print("=" * 70)

    print("\n| Agents | Mean Accuracy | Std Dev | 95% CI | Raw Scores |")
    print("|--------|---------------|---------|--------|------------|")

    for n in agent_counts:
        trial_accs = [sum(trial[n]) / len(trial[n]) for trial in all_trial_results]
        mean_acc = statistics.mean(trial_accs)
        if len(trial_accs) > 1:
            std_acc = statistics.stdev(trial_accs)
            ci = 1.96 * std_acc / (len(trial_accs) ** 0.5)
        else:
            std_acc = 0
            ci = 0
        raw = ", ".join([f"{a:.0%}" for a in trial_accs])
        print(f"| {n} | {mean_acc:.1%} | {std_acc:.1%} | ±{ci:.1%} | {raw} |")

    # Category breakdown
    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY (mean across trials)")
    print("=" * 70)

    categories = ["CRT", "logic", "math"]
    print("\n| Category | n=1 | n=3 | n=5 | n=7 |")
    print("|----------|-----|-----|-----|-----|")

    for cat in categories:
        row = f"| {cat:8} |"
        for n in agent_counts:
            cat_accs = [sum(trial[n][cat]) / len(trial[n][cat]) for trial in all_category_results]
            mean_cat = statistics.mean(cat_accs)
            row += f" {mean_cat:.0%} |"
        print(row)

    # Scaling effect analysis
    print("\n" + "=" * 70)
    print("SCALING EFFECT ANALYSIS")
    print("=" * 70)

    baseline_accs = [sum(trial[1]) / len(trial[1]) for trial in all_trial_results]
    max_agent_accs = [sum(trial[7]) / len(trial[7]) for trial in all_trial_results]

    baseline_mean = statistics.mean(baseline_accs)
    max_mean = statistics.mean(max_agent_accs)
    delta = max_mean - baseline_mean

    print(f"\nBaseline (n=1): {baseline_mean:.1%}")
    print(f"Max agents (n=7): {max_mean:.1%}")
    print(f"Scaling effect: {delta:+.1%} ({'improvement' if delta > 0 else 'degradation' if delta < 0 else 'no change'})")

    # Per-problem error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: Problems with <100% accuracy at any n")
    print("=" * 70)

    # Aggregate across trials for error analysis
    problem_errors = []
    for i, p in enumerate(PROBLEMS):
        error_rates = {}
        for n in agent_counts:
            correct_count = sum(trial[n][i] for trial in all_trial_results)
            error_rates[n] = 1 - (correct_count / N_TRIALS)
        if any(rate > 0 for rate in error_rates.values()):
            problem_errors.append({
                "q": p["q"][:50],
                "category": p["category"],
                "expected": p["a"][0],
                "errors": error_rates
            })

    if problem_errors:
        print("\n| Problem | Cat | Expected | n=1 err | n=3 err | n=5 err | n=7 err |")
        print("|---------|-----|----------|---------|---------|---------|---------|")
        for pe in problem_errors:
            print(f"| {pe['q'][:30]}... | {pe['category']} | {pe['expected']} | {pe['errors'][1]:.0%} | {pe['errors'][3]:.0%} | {pe['errors'][5]:.0%} | {pe['errors'][7]:.0%} |")
    else:
        print("\nNo errors detected across all trials.")

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Total observations: {len(PROBLEMS)} problems × {len(agent_counts)} conditions × {N_TRIALS} trials = {len(PROBLEMS) * len(agent_counts) * N_TRIALS}")
    print(f"Total LLM calls: ~{len(PROBLEMS) * sum(agent_counts) * N_TRIALS}")


if __name__ == "__main__":
    run_experiment()
