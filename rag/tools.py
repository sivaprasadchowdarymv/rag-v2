"""
Tool schemas for the master agent's ReAct loop, plus the safe calculator.

    search_datasheet(query, source)  re-run the specialist retrievers with new keywords
    read_page(page, document)        full text + tables of one page
    calculate(expression)            safe arithmetic (AST walk: no eval, no code execution)

The tools are executed by agents.master.ProToolBox.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: Dict[str, Callable] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "atan": math.atan, "abs": abs, "min": min, "max": max, "round": round,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_calculate(expression: str) -> float:
    expr = (expression or "").replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    if not expr or len(expr) > 200:
        raise ValueError("expression is empty or too long")
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("only numbers, + - * / ** %, parentheses and math functions are allowed")

    return float(ev(tree))


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format, used by OpenRouter)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_datasheet",
            "description": (
                "Search the active documents again. Use specific engineering keywords and symbols "
                "(e.g. 'VOUT output voltage', 'quiescent current IQ', 'pin 3 function'). "
                "Returns numbered evidence to cite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "source": {
                        "type": "string",
                        "enum": ["any", "text", "table", "row", "equation", "figure"],
                        "description": "Restrict to a source type: 'row' = individual table rows/pins.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the full text and tables of one page when excerpts are incomplete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "1-based page number."},
                    "document": {"type": "string", "description": "Part of the file name (only when several "
                                 "documents are active)."},
                },
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate arithmetic exactly, e.g. '(12-5)*1.5' or 'sqrt(2)*3.3'. "
                "Use plain numbers (convert mA to A etc. yourself)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]
