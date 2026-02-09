"""Tests for the shared safe_eval module."""

import pytest
from safe_eval import safe_eval, SafeEvalVisitor


class TestSafeEval:
    """Test safe expression evaluation."""

    def test_arithmetic(self):
        assert safe_eval("1 + 2", {}) == 3
        assert safe_eval("10 - 3", {}) == 7
        assert safe_eval("4 * 5", {}) == 20
        assert safe_eval("10 / 3", {}) == pytest.approx(3.333, rel=0.01)
        assert safe_eval("10 // 3", {}) == 3
        assert safe_eval("10 % 3", {}) == 1
        assert safe_eval("2 ** 3", {}) == 8

    def test_comparison(self):
        assert safe_eval("1 < 2", {}) is True
        assert safe_eval("2 > 3", {}) is False
        assert safe_eval("2 == 2", {}) is True
        assert safe_eval("2 != 3", {}) is True
        assert safe_eval("1 <= 1", {}) is True
        assert safe_eval("2 >= 3", {}) is False

    def test_boolean_ops(self):
        assert safe_eval("True and True", {}) is True
        assert safe_eval("True and False", {}) is False
        assert safe_eval("False or True", {}) is True
        assert safe_eval("False or False", {}) is False

    def test_unary_ops(self):
        assert safe_eval("-5", {}) == -5
        assert safe_eval("not True", {}) is False
        assert safe_eval("not False", {}) is True

    def test_context_variables(self):
        ctx = {"x": 10, "name": "hello"}
        assert safe_eval("x + 5", ctx) == 15
        assert safe_eval("name", ctx) == "hello"

    def test_builtins(self):
        assert safe_eval("len([1, 2, 3])", {}) == 3
        assert safe_eval("min(3, 1, 2)", {}) == 1
        assert safe_eval("max(3, 1, 2)", {}) == 3
        assert safe_eval("abs(-5)", {}) == 5
        assert safe_eval("sum([1, 2, 3])", {}) == 6
        assert safe_eval("int('42')", {}) == 42
        assert safe_eval("float('3.14')", {}) == pytest.approx(3.14)
        assert safe_eval("str(42)", {}) == "42"
        assert safe_eval("bool(1)", {}) is True
        assert safe_eval("repr(42)", {}) == "42"

    def test_container_literals(self):
        assert safe_eval("[1, 2, 3]", {}) == [1, 2, 3]
        assert safe_eval("(1, 2)", {}) == (1, 2)
        assert safe_eval("{'a': 1}", {}) == {'a': 1}

    def test_subscript(self):
        ctx = {"data": [10, 20, 30]}
        assert safe_eval("data[1]", ctx) == 20

    def test_dict_attribute_access(self):
        ctx = {"d": {"key": "value"}}
        assert safe_eval("d.key", ctx) == "value"

    def test_if_expression(self):
        assert safe_eval("1 if True else 2", {}) == 1
        assert safe_eval("1 if False else 2", {}) == 2

    def test_in_operator(self):
        ctx = {"items": [1, 2, 3]}
        assert safe_eval("2 in items", ctx) is True
        assert safe_eval("5 not in items", ctx) is True

    def test_disallowed_name(self):
        with pytest.raises(NameError):
            safe_eval("unknown_var", {})

    def test_disallowed_import(self):
        with pytest.raises((TypeError, NameError)):
            safe_eval("__import__('os')", {})

    def test_disallowed_attribute(self):
        ctx = {"obj": object()}
        with pytest.raises(AttributeError):
            safe_eval("obj.something", ctx)

    def test_disallowed_call(self):
        ctx = {"os_system": exec}
        with pytest.raises(TypeError):
            safe_eval("os_system('echo hi')", ctx)

    def test_constants(self):
        assert safe_eval("True", {}) is True
        assert safe_eval("False", {}) is False
        assert safe_eval("None", {}) is None
        assert safe_eval("42", {}) == 42
        assert safe_eval("'hello'", {}) == "hello"
