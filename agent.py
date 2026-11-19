"""Bounded tool-calling loop with an optional OpenAI backend."""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SYSTEM = (
    "Answer briefly. For shipping questions, call lookup_shipping and use its "
    "result. If no policy is found, say you do not know."
)
POLICIES = {"US": "3–5 business days", "CA": "5–8 business days"}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_shipping",
        "description": "Look up shipping time by two-letter country code.",
        "parameters": {
            "type": "object",
            "properties": {"country": {"type": "string"}},
            "required": ["country"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}]

Model = Callable[[list[dict], list[dict]], dict]


@dataclass
class Result:
    status: str
    answer: Optional[str]
    trace: list[dict]


def dispatch(name: str, raw_arguments: str) -> dict:
    if name != "lookup_shipping":
        return {"error": "unknown_tool"}
    try:
        args = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return {"error": "invalid_arguments"}
    if not isinstance(args, dict) or args.keys() != {"country"}:
        return {"error": "invalid_arguments"}
    country = args["country"]
    if not isinstance(country, str) or len(country) != 2 or not (
        country.isascii() and country.isalpha()
    ):
        return {"error": "invalid_arguments"}
    shipping_time = POLICIES.get(country.upper())
    if shipping_time is None:
        return {"error": "policy_not_found"}
    return {"shipping_time": shipping_time}


def run(prompt: str, model: Model, max_turns: int = 4) -> Result:
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    trace = []
    for turn in range(1, max_turns + 1):
        response = model(messages, TOOLS)
        calls = response.get("tool_calls") or []
        answer = response.get("content")
        if not calls:
            if isinstance(answer, str) and answer.strip():
                return Result("completed", answer, trace)
            return Result("empty_response", None, trace)

        messages.append({"role": "assistant", "content": answer, "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            output = dispatch(name, arguments)
            trace.append({
                "turn": turn, "tool": name, "arguments": arguments, "output": output,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(output),
            })
    return Result("turn_limit", None, trace)


def openai_model(model_name: str, api_key: str) -> Model:
    def complete(messages: list[dict], tools: list[dict]) -> dict:
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": model_name,
                "messages": messages,
                "tools": tools,
                "max_completion_tokens": 512,
                "parallel_tool_calls": False,
            }).encode(),
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                choice = json.load(response)["choices"][0]
        except HTTPError as error:
            raise RuntimeError(f"OpenAI returned HTTP {error.code}") from None
        except (URLError, TimeoutError):
            raise RuntimeError("Could not complete the OpenAI request") from None

        if choice["finish_reason"] not in {"stop", "tool_calls"}:
            raise RuntimeError(f"OpenAI stopped with {choice['finish_reason']}")
        return choice["message"]
    return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--trace", action="store_true", help="print result as JSON")
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        parser.error("set OPENAI_API_KEY")
    try:
        result = run(args.prompt, openai_model(args.model, key))
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    if args.trace:
        print(json.dumps(vars(result), ensure_ascii=False, indent=2))
    else:
        print(result.answer or result.status)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
