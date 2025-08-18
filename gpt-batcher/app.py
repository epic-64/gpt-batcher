import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# -------- Core helpers --------
def hashit(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

async def run_once(client: AsyncOpenAI, prompt: str, model: str, temperature: float, max_output_tokens: int) -> str:
    # For gpt-5 models, we use a different token limit parameter. Thanks, OpenAI!
    token_arg = (
        {"max_completion_tokens": max_output_tokens}
        if model.startswith("gpt-5")
        else {"max_tokens": max_output_tokens}
    )

    # For gpt-5 models, we use minimal reasoning effort by default.
    # For other models, the parameter is not applicable. Thanks, OpenAI!
    reasoning_arg = (
        {"reasoning_effort": "minimal"} if model.startswith("gpt-5") else {}
    )

    # GPT-5 allow ONLY 1.0 temperature, so we set it to 1.0 for those models. Thanks, OpenAI!
    model_aware_temperature = 1.0 if model.startswith("gpt-5") else temperature

    response = await client.chat.completions.create(
        model=model,
        messages=[ChatCompletionUserMessageParam(content=prompt, role="user")],
        temperature=model_aware_temperature,
        **token_arg,
        **reasoning_arg,
    )

    if (choices := response.choices) is None or len(choices) == 0:
        raise ValueError("No choices returned from OpenAI API")

    if (content := choices[0].message.content) is None:
        raise ValueError("No content in the first choice message")

    return content

async def generate(client: AsyncOpenAI, prompt: str, model: str, times: int, temperature: float, max_output_tokens: int):
    tasks = [run_once(client, prompt, model, temperature, max_output_tokens) for _ in range(times)]
    results = await asyncio.gather(*tasks)
    current_date = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    out = {
        "date": current_date,
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "batch_size": times,
        "max_output_tokens": max_output_tokens,
        "results": results,
    }
    file_name = hashit(prompt + model + str(temperature) + str(max_output_tokens))
    file_path = OUTPUTS_DIR.joinpath(f"{current_date}-{file_name}.json")
    file_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path

def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def word_counts(results, top_n: int = 20):
    words = []
    for r in results:
        words.extend(tokenize(r))
    counts = Counter(words)
    return counts.most_common(top_n)

def normalize_response(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def response_counts(results, top_n: int = 20):
    normalized = [normalize_response(r) for r in results]
    counts = Counter(normalized)
    return counts.most_common(top_n)

def plot_word_counts(word_freqs):
    if not word_freqs:
        st.write("No items found")
        return
    words, freqs = zip(*word_freqs)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, freqs)
    ax.set_title("Top Frequencies")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right")
    st.pyplot(fig)

# -------- Streamlit UI --------
st.title("GPT Batch Generator + Visualizer")

st.header("Generate new batch")
with st.form("generate_form"):
    api_key = st.text_input("OpenAI API Key", type="password")
    prompt = st.text_area("Prompt", "Give me any book title (respond with just the name)")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-5-2025-08-07", "gpt-3.5-turbo"])
    times = st.number_input("Times (how many times to run the prompt)", min_value=1, max_value=100, value=50)
    temperature = st.slider("Temperature (randomness when picking a word from the rankings)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
    max_output_tokens = st.number_input(
        "Max output tokens (hard cap per response). This is to save you from unexpected costs.",
        min_value=1,
        max_value=100,
        value=25,
        step=1,
        help="Upper bound for tokens the model may generate per response.",
    )

    projected_max = times * max_output_tokens
    st.caption(f"Projected worst-case output tokens this run: **{projected_max:,}**")

    submitted = st.form_submit_button("Run batch")
    if submitted:
        if not api_key:
            st.error("Please enter your OpenAI API Key before running the batch.")
        else:
            with st.spinner("Generating..."):
                client = AsyncOpenAI(api_key=api_key)
                file_path = asyncio.run(
                    generate(client, prompt, model, int(times), float(temperature), int(max_output_tokens))
                )
            st.success(f"Results saved to {file_path}")

st.header("Visualize existing batch")
files = sorted(OUTPUTS_DIR.glob("*.json"), reverse=True)
if not files:
    st.write("No files in outputs/ yet.")
else:
    file_labels = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            run_date = data.get("date", "?")
            model_name = data.get("model", "?")
            temperature_val = data.get("temperature", "?")
            batch_size = data.get("batch_size", "?")
            prompt_text = (data.get("prompt") or "").replace("\n", " ")
            prompt_preview = prompt_text[:60] + ("…" if len(prompt_text) > 60 else "")
            label = f"{model_name} · temp={temperature_val} · {batch_size} · {prompt_preview}"
        except Exception:
            label = f"{f.name} | <error reading>"
        file_labels.append(label)

    idx = st.selectbox("Choose file", range(len(files)), format_func=lambda i: file_labels[i])
    selected_file = files[idx]

    # Load data for visualization
    data = json.loads(selected_file.read_text(encoding="utf-8"))
    results = data.get("results", [])

    # ---- Controls ----
    top_n = st.slider("Display Top", 5, 50, 50)
    grouping_mode = st.radio("Group results by:", ["Responses", "Words"], horizontal=True)

    # ---- Metadata ----
    prompt_text = data.get("prompt", None)
    model_used = data.get("model", None)
    temperature_used = data.get("temperature", None)
    run_date = data.get("date", None)
    batch_size = data.get("batch_size", None)
    max_tokens_used = data.get("max_output_tokens", None)

    # compute group count depending on grouping mode
    if grouping_mode == "Words":
        all_words = []
        for r in results:
            all_words.extend(tokenize(r))
        group_count = len(set(all_words))
    elif grouping_mode == "Responses":
        normalized = [normalize_response(r) for r in results]
        group_count = len(set(normalized))
    else:
        group_count = 0

    separator = "&nbsp;&nbsp; | &nbsp;&nbsp;"
    st.subheader("Batch Info")
    st.markdown(
        f"**Date:** {run_date} {separator} "
        f"**Max Tokens:** {max_tokens_used}"
    )
    st.markdown(
        f"**Model:** `{model_used}` {separator} "
        f"**Temperature:** {temperature_used} {separator} "
        f"**Batch Size:** {batch_size} {separator} "
        f"**Display Top:** {top_n} {separator} "
        f"**Groups:** {group_count}"
    )
    st.markdown(f"**Prompt:** {prompt_text}")

    # ---- Frequencies ----
    match grouping_mode:
        case "Words": frequencies = word_counts(results, top_n)
        case "Responses": frequencies = response_counts(results, top_n)
        case _: frequencies = []

    plot_word_counts(frequencies)

    # ---- All responses ----
    st.subheader("All Responses")
    if not results:
        st.write("No responses in this file.")
    else:
        st.table({"Content": results})
