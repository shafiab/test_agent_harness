"""Offline regression evals. Run with python3 -m unittest -v evals."""

import json
import unittest

from agent import run


def tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class AgentEvals(unittest.TestCase):
    def test_direct_answer(self):
        def model(messages, tools):
            self.assertEqual(messages[-1]["content"], "Hello")
            return {"content": "Hi!"}

        result = run("Hello", model)
        self.assertEqual((result.status, result.answer, result.trace),
                         ("completed", "Hi!", []))

    def test_answer_uses_shipping_tool_result(self):
        call = tool_call("lookup_shipping", '{"country":"ca"}')

        def model(messages, tools):
            self.assertEqual(tools[0]["function"]["name"], "lookup_shipping")
            if messages[-1]["role"] == "user":
                return {"tool_calls": [call]}
            self.assertEqual(messages[-2]["tool_calls"], [call])
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertEqual(messages[-1]["tool_call_id"], "call_1")
            value = json.loads(messages[-1]["content"])["shipping_time"]
            return {"content": f"Shipping to Canada takes {value}."}

        result = run("How long does shipping to Canada take?", model)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "Shipping to Canada takes 5–8 business days.")
        self.assertEqual(len(result.trace), 1)

    def test_unknown_country(self):
        def model(messages, tools):
            if messages[-1]["role"] == "user":
                return {"tool_calls": [
                    tool_call("lookup_shipping", '{"country":"GB"}'),
                ]}
            self.assertEqual(json.loads(messages[-1]["content"]),
                             {"error": "policy_not_found"})
            return {"content": "I don’t know the shipping time."}

        self.assertEqual(run("Shipping to GB?", model).answer,
                         "I don’t know the shipping time.")

    def test_disallowed_tool_and_invalid_arguments(self):
        cases = [
            (tool_call("unlisted", "{}"), "unknown_tool"),
            (tool_call("lookup_shipping", "{"), "invalid_arguments"),
            (tool_call("lookup_shipping", '{"country":42}'), "invalid_arguments"),
            (tool_call("lookup_shipping", '{"country":"US","extra":true}'),
             "invalid_arguments"),
        ]
        for call, error in cases:
            with self.subTest(call=call):
                def model(messages, tools):
                    if messages[-1]["role"] == "user":
                        return {"tool_calls": [call]}
                    self.assertEqual(json.loads(messages[-1]["content"]),
                                     {"error": error})
                    return {"content": "I couldn’t look that up."}

                result = run("Look up shipping", model)
                self.assertEqual(result.status, "completed")
                self.assertEqual(result.trace[0]["output"], {"error": error})

    def test_turn_limit_stops_repeated_calls(self):
        turns = []

        def model(messages, tools):
            turns.append(len(messages))
            return {"tool_calls": [tool_call("lookup_shipping", '{"country":"US"}',
                                            f"call_{len(turns)}")]}

        result = run("Keep looking up shipping", model, max_turns=2)
        self.assertEqual((result.status, result.answer), ("turn_limit", None))
        self.assertEqual(turns, [2, 4])
        self.assertEqual([event["turn"] for event in result.trace], [1, 2])

    def test_empty_response(self):
        result = run("Hello", lambda messages, tools: {"content": " "})
        self.assertEqual((result.status, result.answer), ("empty_response", None))


if __name__ == "__main__":
    unittest.main()
