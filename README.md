# Minimal agent harness

A small Python tool-calling agent with one read-only tool. The model can answer
directly or request a shipping lookup. The harness validates arguments, runs the
lookup, returns the result to the model, and stops after at most four model calls.

## Run the offline evals

Requires Python 3.9 or later. The harness and evals use the standard library.

```sh
python3 -m unittest -v evals
```

The evals use scripted model responses to check direct answers, tool result
handoff, missing policies, invalid tool calls, the turn limit, and empty
responses. They exercise harness behavior deterministically. Model quality can
be explored separately with the live CLI.

## Try a live model

Set `OPENAI_API_KEY` in your shell, then run:

```sh
python3 agent.py "How long does shipping to Canada take?" --trace
```

`--trace` prints the answer, stop status, and tool calls as JSON. Use
`--model MODEL_ID` to override the default model. The shipping policies are fixture
data. The [OpenAI Chat Completions API reference](https://platform.openai.com/docs/api-reference/chat/create)
describes the model adapter's wire format.

To try another backend, pass a callable that accepts `(messages, tools)` and
returns an assistant message to `run()` in `agent.py`. Add a scripted case in
`evals.py` to cover the behavior you want to keep.
