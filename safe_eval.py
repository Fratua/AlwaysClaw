"""
Shared safe expression evaluator (replaces eval()).

Provides an AST-based visitor that only allows safe operations:
comparisons, boolean ops, arithmetic, string ops, attribute access on
dict values, subscripts, and whitelisted builtins.

Used by e2e_loop_core.py and self_driven_loop/proactive_trigger.py.
"""

import ast
import operator
from typing import Any, Dict

SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Not: operator.not_,
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}

SAFE_BUILTINS = {
    'len': len, 'min': min, 'max': max, 'abs': abs,
    'sum': sum, 'any': any, 'all': all, 'bool': bool,
    'int': int, 'float': float, 'str': str, 'repr': repr,
    'True': True, 'False': False, 'None': None,
}


class SafeEvalVisitor(ast.NodeVisitor):
    """Walk an AST and evaluate only safe operations."""

    def __init__(self, context: Dict[str, Any]):
        self.context = {**SAFE_BUILTINS, **context}

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    # Python 3.7 compat
    visit_Num = visit_Constant
    visit_Str = visit_Constant
    visit_NameConstant = visit_Constant

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.context:
            return self.context[node.id]
        raise NameError(f"Name '{node.id}' is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        if isinstance(value, dict):
            return value.get(node.attr)
        raise AttributeError(f"Attribute access on {type(value).__name__} not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        idx = self.visit(node.slice)
        return value[idx]

    def visit_Index(self, node: ast.Index) -> Any:
        # Python <3.9 wraps subscript values in Index
        return self.visit(node.value)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = self.visit(v)
                if not result:
                    return result
            return result
        else:  # Or
            result = False
            for v in node.values:
                result = self.visit(v)
                if result:
                    return result
            return result

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Operator {type(node.op).__name__} not allowed")
        return op_func(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unary operator {type(node.op).__name__} not allowed")
        return op_func(self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_func = SAFE_OPS.get(type(op))
            if op_func is None:
                raise TypeError(f"Comparator {type(op).__name__} not allowed")
            right = self.visit(comparator)
            if not op_func(left, right):
                return False
            left = right
        return True

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        if not callable(func):
            raise TypeError(f"{func!r} is not callable")
        if func not in SAFE_BUILTINS.values():
            raise TypeError(f"Calling {func!r} is not allowed")
        args = [self.visit(a) for a in node.args]
        return func(*args)

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in node.elts)

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    def generic_visit(self, node: ast.AST) -> Any:
        raise TypeError(f"AST node type {type(node).__name__} not allowed")


def safe_eval(expression: str, context: Dict[str, Any]) -> Any:
    """Safely evaluate a simple expression against a context dict.

    Only allows: comparisons, boolean ops, arithmetic, string ops,
    attribute access on dict values, subscripts, and whitelisted builtins.
    """
    tree = ast.parse(expression, mode='eval')
    visitor = SafeEvalVisitor(context)
    return visitor.visit(tree)
