
import os
import json
import ast
import operator
import urllib.request
import urllib.error
from datetime import datetime

# ==========================================
# QntaAgent v0.1
# Experimental AI Agent
# ==========================================

APP_NAME = "QntaAgent"
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

MAX_STEPS = 4

SYSTEM_PROMPT = """
You are QntaAgent, an experimental AI agent.

You can answer questions and use registered tools.

You MUST respond using valid JSON.

For a normal answer:
{
  "type": "final",
  "message": "Your answer"
}

For a tool call:
{
  "type": "tool",
  "name": "calculator",
  "arguments": {
    "expression": "25 * 4"
  }
}

Available tools:

1. calculator
   Arguments: expression (string)

2. current_time
   Arguments: {}

Rules:
- Never invent tool results.
- Only request registered tools.
- Use tools when necessary.
- Do not claim a task was completed unless it was.
- Do not attempt to execute arbitrary code.
- Return one JSON object per response.
"""

# ==========================================
# SAFE CALCULATOR
# ==========================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate(expression):

    if len(expression) > 100:
        raise ValueError("Expression too long")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only numbers allowed")
            return node.value

        if isinstance(node, ast.BinOp):

            op = type(node.op)

            if op not in OPERATORS:
                raise ValueError("Unsupported operation")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if op is ast.Pow and abs(right) > 10:
                raise ValueError("Exponent too large")

            result = OPERATORS[op](left, right)

            if abs(result) > 10**100:
                raise ValueError("Result too large")

            return result

        if isinstance(node, ast.UnaryOp):

            op = type(node.op)

            if op not in UNARY:
                raise ValueError("Unsupported operation")

            return UNARY[op](evaluate(node.operand))

        raise ValueError("Invalid expression")

    return evaluate(tree)


# ==========================================
# TOOL REGISTRY
# ==========================================

def current_time():

    return datetime.now().astimezone().isoformat()


TOOLS = {
    "calculator": calculate,
    "current_time": current_time,
}


def execute_tool(name, arguments):

    if name not in TOOLS:
        return {"error": "Tool not registered"}

    try:

        if name == "calculator":
            result = TOOLS[name](
                arguments["expression"]
            )
        else:
            result = TOOLS[name]()

        return {"result": result}

    except Exception as error:
        return {"error": str(error)}


# ==========================================
# AI MODEL CONNECTION
# ==========================================

def ask_model(messages):

    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured"
        )

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1200
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://qntatechnologies.onrender.com",
            "X-Title": APP_NAME
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request, timeout=60
    ) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    return result["choices"][0]["message"]["content"]


# ==========================================

# ==========================================
# AGENT ENGINE v0.1.1
# ==========================================

def parse_agent_response(raw):
    """Parse plain or Markdown-wrapped JSON."""

    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
        else:
            raw = "\n".join(lines[1:]).strip()

    try:
        response = json.loads(raw)

        if not isinstance(response, dict):
            raise ValueError("Expected a JSON object")

        return response

    except (json.JSONDecodeError, ValueError) as error:
        print("\n[DIAGNOSTIC] JSON parsing failed:", error)
        print("[DIAGNOSTIC] Raw response:", repr(raw))
        return None


def run_agent(user_input, history):

    messages = history + [
        {"role": "user", "content": user_input}
    ]

    messages.insert(
        0,
        {"role": "system", "content": SYSTEM_PROMPT}
    )

    for step in range(MAX_STEPS):

        print(f"\n[AGENT] Step {step + 1}/{MAX_STEPS}")

        try:
            raw = ask_model(messages)

        except Exception as error:
            print("[ERROR] Model request failed:", error)
            return "AI provider request failed."

        response = parse_agent_response(raw)

        if response is None:
            return "Invalid AI response. See diagnostics above."

        response_type = response.get("type")

        if response_type == "final":

            answer = response.get("message")

            if not isinstance(answer, str):
                return "Invalid final response format."

            history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": raw}
            ])

            return answer

        if response_type == "tool":

            name = response.get("name")
            arguments = response.get("arguments", {})

            if not isinstance(arguments, dict):
                return "Invalid tool arguments."

            print(f"[TOOL] Requested: {name}")
            print(f"[TOOL] Arguments: {arguments}")

            result = execute_tool(name, arguments)

            print("[TOOL] Result:", result)

            messages.append({
                "role": "assistant",
                "content": raw
            })

            messages.append({
                "role": "user",
                "content": (
                    "Tool execution result: "
                    + json.dumps(result)
                )
            })

        else:
            print("[DIAGNOSTIC] Unknown response type:", response_type)
            print("[DIAGNOSTIC] Raw response:", repr(raw))
            return "Unknown agent response type."

    return "Agent reached its maximum step limit."


# MAIN INTERFACE
# ==========================================

def main():

    print("=" * 42)
    print("       QntaAgent v0.1")
    print("   Experimental Agent Prototype")
    print("=" * 42)

    history = []

    while True:

        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ("exit", "quit"):
                print("QntaAgent: Shutting down.")
                break

            if not user_input:
                continue

            answer = run_agent(
                user_input, history
            )

            print("\nQntaAgent:", answer)

        except KeyboardInterrupt:
            print("\nQntaAgent: Goodbye.")
            break

        except Exception as error:
            print("\nAgent error:", error)


if __name__ == "__main__":
    main()
