"""
Ablation study: How does the number of agents affect Flow-of-Options accuracy?

Flow-of-Options = multiple LLM "agents" propose answers, then vote/aggregate.
We test: does more agents = better results? Is there a sweet spot?
"""
import sys
sys.path.insert(0, "/home/OSPIL/Hackathon-Scientific-Discovery")

from hackathon_science.utils import call_llm
import random

# Mix of easy and tricky reasoning problems
PROBLEMS = [
    # Classic cognitive reflection test
    {"q": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost in cents?", "a": "5"},
    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?", "a": "5"},
    {"q": "In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half the lake?", "a": "47"},
    # Trickier logic
    {"q": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?", "a": "no"},
    {"q": "All bloops are razzies. All razzies are lazzies. Are all bloops definitely lazzies?", "a": "yes"},
    {"q": "Some gribs are zorps. All zorps are flims. Are some gribs definitely flims?", "a": "yes"},
    {"q": "No wugs are tims. Some tims are pobs. Are some wugs definitely not pobs?", "a": "no"},
    # Math edge cases
    {"q": "What is 17 x 24?", "a": "408"},
    {"q": "If you have 3 apples and take away 2, how many do YOU have?", "a": "2"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "a": "9"},
]

MODEL = "global.anthropic.claude-sonnet-4-6"


def get_single_answer(problem: str) -> str:
    """Get one agent's answer."""
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": f"Answer with just one word or number, no formatting: {problem}"}]}],
        model_id=MODEL
    )
    text = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip().lower()
    # Strip markdown formatting
    text = text.replace("*", "").replace("`", "")
    # Extract just the key part
    for word in text.split():
        word = word.strip(".,!?")
        if word in ["yes", "no", "5", "47", "true", "false", "valid", "invalid"]:
            return word
    return text.split()[0] if text else ""


def flow_of_options(problem: str, n_agents: int) -> str:
    """Run Flow-of-Options: n agents propose, then majority vote."""
    votes = []
    for _ in range(n_agents):
        answer = get_single_answer(problem)
        votes.append(answer)

    # Majority vote
    from collections import Counter
    counts = Counter(votes)
    winner = counts.most_common(1)[0][0]
    return winner, votes


def normalize_answer(answer: str, expected: str) -> bool:
    """Check if answer matches expected (with some flexibility)."""
    answer = answer.lower().strip()
    expected = expected.lower().strip()

    # Direct match
    if answer == expected:
        return True
    # Handle yes/valid, no/invalid
    if expected == "yes" and answer in ["yes", "valid", "true"]:
        return True
    if expected == "no" and answer in ["no", "invalid", "false", "cannot"]:
        return True
    # Handle numbers that might have extra text
    if expected.isdigit() and expected in answer:
        return True
    return False


def run_experiment():
    """Run ablation across different agent counts."""
    agent_counts = [1, 3, 5, 7]
    results = {n: [] for n in agent_counts}

    print("=" * 60)
    print("FLOW-OF-OPTIONS ABLATION: NUMBER OF AGENTS")
    print("=" * 60)

    for problem in PROBLEMS:
        print(f"\nQ: {problem['q'][:60]}...")
        print(f"Expected: {problem['a']}")

        for n in agent_counts:
            answer, votes = flow_of_options(problem["q"], n)
            correct = normalize_answer(answer, problem["a"])
            results[n].append(correct)
            print(f"  n={n}: {answer} (votes: {votes}) {'✓' if correct else '✗'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for n in agent_counts:
        acc = sum(results[n]) / len(results[n])
        print(f"n={n} agents: {acc:.0%} accuracy ({sum(results[n])}/{len(results[n])})")

    print("\n" + "=" * 60)
    print("RAW DATA FOR PAPER")
    print("=" * 60)
    print("| Agents | Correct | Total | Accuracy |")
    print("|--------|---------|-------|----------|")
    for n in agent_counts:
        correct = sum(results[n])
        total = len(results[n])
        acc = correct / total
        print(f"| {n} | {correct} | {total} | {acc:.0%} |")


if __name__ == "__main__":
    run_experiment()
