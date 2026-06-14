from __future__ import annotations

from dataclasses import dataclass

from qfr.config import TokenConfig


@dataclass(frozen=True, slots=True)
class TokenSpec:
    name: str
    kind: str


class Vocabulary:
    def __init__(self, specs: list[TokenSpec]) -> None:
        self.specs = specs
        self.token_to_id = {spec.name: idx for idx, spec in enumerate(specs)}

    def __len__(self) -> int:
        return len(self.specs)

    def spec(self, token_id: int) -> TokenSpec:
        return self.specs[token_id]

    def name(self, token_id: int) -> str:
        return self.spec(token_id).name

    def id_of(self, name: str) -> int:
        return self.token_to_id[name]

    @property
    def begin_id(self) -> int:
        return self.id_of("BEG")

    @property
    def sep_id(self) -> int:
        return self.id_of("SEP")


def build_vocabulary(cfg: TokenConfig) -> Vocabulary:
    specs = [TokenSpec("BEG", "begin"), TokenSpec("SEP", "sep")]
    specs.extend(TokenSpec(name, "feature") for name in cfg.features)
    specs.extend(TokenSpec(str(value), "constant") for value in cfg.constants)
    specs.extend(TokenSpec(f"{value}d", "window") for value in cfg.windows)
    specs.extend(
        [
            TokenSpec("Abs", "cross_unary"),
            TokenSpec("Log", "cross_unary"),
            TokenSpec("Add", "cross_binary"),
            TokenSpec("Sub", "cross_binary"),
            TokenSpec("Mul", "cross_binary"),
            TokenSpec("Div", "cross_binary"),
            TokenSpec("Larger", "cross_binary"),
            TokenSpec("Smaller", "cross_binary"),
            TokenSpec("Ref", "ts_unary"),
            TokenSpec("Mean", "ts_unary"),
            TokenSpec("Median", "ts_unary"),
            TokenSpec("Sum", "ts_unary"),
            TokenSpec("Std", "ts_unary"),
            TokenSpec("Var", "ts_unary"),
            TokenSpec("Max", "ts_unary"),
            TokenSpec("Min", "ts_unary"),
            TokenSpec("Mad", "ts_unary"),
            TokenSpec("Delta", "ts_unary"),
            TokenSpec("WMA", "ts_unary"),
            TokenSpec("EMA", "ts_unary"),
            TokenSpec("Cov", "ts_binary"),
            TokenSpec("Corr", "ts_binary"),
        ]
    )
    return Vocabulary(specs)
