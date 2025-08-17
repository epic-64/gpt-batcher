import asyncio
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from openai import AsyncOpenAI

st.sidebar.header("API Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

client = AsyncOpenAI(api_key=api_key)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


# -------- Core helpers --------
def hashit(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


async def run_once(prompt: str, model: str) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    if (choices := response.choices) is None or len(choices) == 0:
        raise ValueError("No choices returned from OpenAI API")

    if (content := choices[0].message.content) is None:
        raise ValueError("No content in the first choice message")

    return content


async def generate(prompt: str, model: str, times: int):
    tasks = [run_once(prompt, model) for _ in range(times)]
    results = await asyncio.gather(*tasks)
    current_date = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    out = {"date": current_date, "prompt": prompt, "model": model, "batch_size": times, "results": results}
    file_name = hashit(prompt + model)
    file_path = OUTPUTS_DIR.joinpath(f"{current_date}-{file_name}.json")
    file_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def load_results(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results", [])


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def word_counts(results, top_n: int = 20):
    words = []
    for r in results:
        words.extend(tokenize(r))
    counts = Counter(words)
    return counts.most_common(top_n)


def plot_word_counts(word_freqs):
    if not word_freqs:
        st.write("No words found")
        return
    words, freqs = zip(*word_freqs)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, freqs)
    ax.set_title("Top Words")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right")
    st.pyplot(fig)


# -------- Streamlit UI --------
st.title("GPT Batch Generator + Visualizer")

st.header("Generate new batch")
with st.form("generate_form"):
    prompt = st.text_area("Prompt", "Give me any book title (respond with just the name")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-5-2025-08-07", "gpt-3.5-turbo"])
    times = st.number_input("Times", min_value=1, max_value=100, value=50)
    submitted = st.form_submit_button("Run batch")
    if submitted:
        with st.spinner("Generating..."):
            file_path = asyncio.run(generate(prompt, model, times))
        st.success(f"Results saved to {file_path}")


st.header("Visualize existing batch")
files = sorted(OUTPUTS_DIR.glob("*.json"), reverse=True)
if not files:
    st.write("No files in outputs/ yet.")
else:
    # Build display labels with prompt+model
    file_labels = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            prompt = (data.get("prompt") or "").replace("\n", " ")
            prompt_preview = prompt[:100] + ("…" if len(prompt) > 100 else "")
            model = data.get("model", "?")
            label = f"{f.name} | {model} | {prompt_preview}"
        except Exception:
            label = f"{f.name} | <error reading>"
        file_labels.append(label)

    # Dropdown shows friendly label, returns file index
    idx = st.selectbox("Choose file", range(len(files)), format_func=lambda i: file_labels[i])
    selected_file = files[idx]

    top_n = st.slider("Top N words", 5, 50, 50)
    if st.button("Visualize"):
        data = json.loads(selected_file.read_text(encoding="utf-8"))
        results = data.get("results", [])

        # ---- Metadata ----
        prompt_text = data.get("prompt", None)
        model_used = data.get("model", None)
        run_date = data.get("date", None)
        batch_size = data.get("batch_size", None)

        # Unique word count across all responses
        all_words = []
        for r in results:
            all_words.extend(tokenize(r))
        unique_words = len(set(all_words))

        separator = "&nbsp;&nbsp; | &nbsp;&nbsp;"
        st.subheader("Batch Info")
        st.markdown(f"**Date:** {run_date} ")
        st.markdown(
            f"**Model:** `{model_used}` {separator} "
            f"**Batch Size:** {batch_size} {separator} "
            f"**Top N:** {top_n} {separator} "
            f"**Unique Words:** {unique_words}"
        )

        # Full prompt (can get long, so keep it on its own line)
        st.markdown(f"**Prompt:** {prompt_text}")

        # ---- Word frequency chart ----
        frequencies = word_counts(results, top_n)
        plot_word_counts(frequencies)

        # ---- All responses ----
        st.subheader("All Responses")
        if not results:
            st.write("No responses in this file.")
        else:
            st.table({
                "Response #": list(range(1, len(results) + 1)),
                "Content": results,
            })