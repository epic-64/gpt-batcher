import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


# -------- Core helpers --------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


async def run_once(prompt: str, model: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


async def generate(prompt: str, model: str, times: int):
    tasks = [run_once(prompt, model) for _ in range(times)]
    results = await asyncio.gather(*tasks)

    out = {"prompt": prompt, "model": model, "results": results}

    current_date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = sha256(prompt + model)
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
    prompt = st.text_area("Prompt", "Say hi like a pirate")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"])
    times = st.number_input("Times", min_value=1, max_value=50, value=5)
    submitted = st.form_submit_button("Run batch")
    if submitted:
        with st.spinner("Generating..."):
            file_path = asyncio.run(generate(prompt, model, times))
        st.success(f"Results saved to {file_path}")


st.header("Visualize existing batch")
files = sorted(OUTPUTS_DIR.glob("*.json"))
if not files:
    st.write("No files in outputs/ yet.")
else:
    selected_file = st.selectbox("Choose file", files)
    top_n = st.slider("Top N words", 5, 50, 20)
    if st.button("Visualize"):
        results = load_results(selected_file)
        freqs = word_counts(results, top_n)
        plot_word_counts(freqs)