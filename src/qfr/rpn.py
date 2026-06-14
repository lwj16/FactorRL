from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from qfr.data import MarketData
from qfr.tokens import TokenSpec, Vocabulary


@dataclass(frozen=True, slots=True)
class WindowLiteral:
    days: int


@dataclass(frozen=True, slots=True)
class Expression:
    kind: str
    value: str | float | None = None
    args: tuple[object, ...] = ()

    def render(self) -> str:
        if self.kind == "feature":
            return str(self.value)
        if self.kind == "constant":
            return str(self.value)
        if self.kind in {"cross_unary", "ts_unary"}:
            rendered_args = ", ".join(_render_arg(arg) for arg in self.args)
            return f"{self.value}({rendered_args})"
        rendered_args = ", ".join(_render_arg(arg) for arg in self.args)
        return f"{self.value}({rendered_args})"

    def evaluate(self, data: MarketData) -> pd.DataFrame:
        template = next(iter(data.features.values()))
        if self.kind == "feature":
            return data.features[str(self.value)]
        if self.kind == "constant":
            filled = np.full(template.shape, float(self.value), dtype=float)
            return pd.DataFrame(filled, index=template.index, columns=template.columns)

        evaluated = [
            arg.evaluate(data) if isinstance(arg, Expression) else arg for arg in self.args
        ]
        name = str(self.value)
        if self.kind == "cross_unary":
            source = evaluated[0]
            if name == "Abs":
                return source.abs()
            if name == "Log":
                if (source <= 0).any().any():
                    raise ValueError("Log received non-positive values.")
                return np.log(source)
        if self.kind == "cross_binary":
            left, right = evaluated
            if name == "Add":
                return left + right
            if name == "Sub":
                return left - right
            if name == "Mul":
                return left * right
            if name == "Div":
                safe = right.where(right.abs() > 1e-12)
                return left / safe
            if name == "Larger":
                return left.where(left >= right, right)
            if name == "Smaller":
                return left.where(left <= right, right)
        if self.kind == "ts_unary":
            source, window = evaluated
            days = _window_days(window)
            if name == "Ref":
                return source.shift(days)
            if name == "Mean":
                return source.rolling(days, min_periods=days).mean()
            if name == "Median":
                return source.rolling(days, min_periods=days).median()
            if name == "Sum":
                return source.rolling(days, min_periods=days).sum()
            if name == "Std":
                return source.rolling(days, min_periods=days).std()
            if name == "Var":
                return source.rolling(days, min_periods=days).var()
            if name == "Max":
                return source.rolling(days, min_periods=days).max()
            if name == "Min":
                return source.rolling(days, min_periods=days).min()
            if name == "Mad":
                return source.rolling(days, min_periods=days).apply(
                    lambda values: float(np.mean(np.abs(values - np.mean(values)))),
                    raw=True,
                )
            if name == "Delta":
                return source - source.shift(days)
            if name == "WMA":
                weights = np.arange(1, days + 1, dtype=float)
                return source.rolling(days, min_periods=days).apply(
                    lambda values: float(np.dot(values, weights) / weights.sum()),
                    raw=True,
                )
            if name == "EMA":
                return source.ewm(span=days, min_periods=days, adjust=False).mean()
        if self.kind == "ts_binary":
            left, right, window = evaluated
            days = _window_days(window)
            if name == "Cov":
                return left.rolling(days, min_periods=days).cov(right)
            if name == "Corr":
                return left.rolling(days, min_periods=days).corr(right)
        raise ValueError(f"Unsupported expression kind: {self.kind} / {self.value}")


def _render_arg(arg: object) -> str:
    if isinstance(arg, Expression):
        return arg.render()
    if isinstance(arg, WindowLiteral):
        return f"{arg.days}d"
    return str(arg)


def _window_days(window: object) -> int:
    if not isinstance(window, WindowLiteral):
        raise ValueError("Time-series operators require a window token.")
    return window.days


def apply_token_to_stack(stack: list[str], spec: TokenSpec) -> list[str] | None:
    new_stack = list(stack)
    if spec.kind in {"feature", "constant"}:
        new_stack.append("value")
        return new_stack
    if spec.kind == "window":
        new_stack.append("window")
        return new_stack
    if spec.kind == "cross_unary":
        if len(new_stack) < 1 or new_stack[-1] != "value":
            return None
        return new_stack
    if spec.kind == "cross_binary":
        if len(new_stack) < 2 or new_stack[-2:] != ["value", "value"]:
            return None
        new_stack[-2:] = ["value"]
        return new_stack
    if spec.kind == "ts_unary":
        if len(new_stack) < 2 or new_stack[-2:] != ["value", "window"]:
            return None
        new_stack[-2:] = ["value"]
        return new_stack
    if spec.kind == "ts_binary":
        if len(new_stack) < 3 or new_stack[-3:] != ["value", "value", "window"]:
            return None
        new_stack[-3:] = ["value"]
        return new_stack
    return None


def parse_expression(token_ids: Iterable[int], vocab: Vocabulary) -> Expression:
    stack: list[Expression | WindowLiteral] = []
    for token_id in token_ids:
        spec = vocab.spec(token_id)
        if spec.kind in {"begin", "sep"}:
            continue
        if spec.kind == "feature":
            stack.append(Expression(kind="feature", value=spec.name))
            continue
        if spec.kind == "constant":
            stack.append(Expression(kind="constant", value=float(spec.name)))
            continue
        if spec.kind == "window":
            stack.append(WindowLiteral(days=int(spec.name[:-1])))
            continue
        if spec.kind == "cross_unary":
            arg = stack.pop()
            if not isinstance(arg, Expression):
                raise ValueError("Invalid unary cross expression.")
            stack.append(Expression(kind="cross_unary", value=spec.name, args=(arg,)))
            continue
        if spec.kind == "cross_binary":
            right = stack.pop()
            left = stack.pop()
            if not isinstance(left, Expression) or not isinstance(right, Expression):
                raise ValueError("Invalid binary cross expression.")
            stack.append(Expression(kind="cross_binary", value=spec.name, args=(left, right)))
            continue
        if spec.kind == "ts_unary":
            window = stack.pop()
            source = stack.pop()
            if not isinstance(source, Expression) or not isinstance(window, WindowLiteral):
                raise ValueError("Invalid unary time-series expression.")
            stack.append(Expression(kind="ts_unary", value=spec.name, args=(source, window)))
            continue
        if spec.kind == "ts_binary":
            window = stack.pop()
            right = stack.pop()
            left = stack.pop()
            if not isinstance(left, Expression) or not isinstance(right, Expression) or not isinstance(window, WindowLiteral):
                raise ValueError("Invalid binary time-series expression.")
            stack.append(Expression(kind="ts_binary", value=spec.name, args=(left, right, window)))
            continue
        raise ValueError(f"Unexpected token: {spec.name}")
    if len(stack) != 1 or not isinstance(stack[0], Expression):
        raise ValueError("Sequence did not reduce to one expression.")
    return stack[0]
