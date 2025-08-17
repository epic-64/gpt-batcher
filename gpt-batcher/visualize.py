import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results", [])


def tokenize(text: str) -> list[str]:
    # Lowercase, split on non-letters
    return re.findall(r"\b\w+\b", text.lower())


def visualize_word_counts(results: list[str], top_n: int = 20):
    words = []
    for r in results:
        words.extend(tokenize(r))

    counts = Counter(words)
    most_common = counts.most_common(top_n)

    if not most_common:
        print("No words found.")
        return

    words, freqs = zip(*most_common)

    plt.figure(figsize=(10, 6))
    plt.bar(words, freqs)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} Words Across All Results")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize word frequencies from GPT batch output")
    parser.add_argument("file", type=Path, help="Path to JSON output file")
    parser.add_argument("--top", type=int, default=20, help="How many top words to show")
    args = parser.parse_args()

    results = load_results(args.file)
    visualize_word_counts(results, args.top)


if __name__ == "__main__":
    main()
