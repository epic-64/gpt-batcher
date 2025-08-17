import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env if present
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


async def run_once(prompt: str, model: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


async def main():
    parser = argparse.ArgumentParser(description="Batch OpenAI requests")
    parser.add_argument("--prompt", "-p", required=True, help="The prompt to send")
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g. gpt-4o-mini)")
    parser.add_argument("--times", "-n", type=int, required=True, help="How many times to run the prompt")

    args = parser.parse_args()

    tasks = [run_once(args.prompt, args.model) for _ in range(args.times)]
    results = await asyncio.gather(*tasks)

    out = {
        "prompt": args.prompt,
        "model": args.model,
        "results": results,
    }

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    current_date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = sha256(args.prompt + args.model)
    file_path = outputs_dir.joinpath(f"{current_date}-{file_name}.json")
    Path(file_path).write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Wrote {len(results)} results to {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
