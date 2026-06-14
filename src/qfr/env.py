from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfr.rpn import apply_token_to_stack
from qfr.tokens import Vocabulary


@dataclass(slots=True)
class FormulaState:
    token_ids: list[int]
    type_stack: list[str]
    done: bool = False


class FormulaEnv:
    def __init__(self, vocab: Vocabulary, max_length: int) -> None:
        self.vocab = vocab
        self.max_length = max_length

    def reset(self) -> FormulaState:
        return FormulaState(token_ids=[self.vocab.begin_id], type_stack=[], done=False)

    def legal_action_mask(self, state: FormulaState) -> np.ndarray:
        mask = np.zeros(len(self.vocab), dtype=bool)
        content_length = len(state.token_ids) - 1
        if state.done:
            return mask
        for token_id in range(len(self.vocab)):
            spec = self.vocab.spec(token_id)
            if spec.kind == "begin":
                continue
            if spec.kind == "sep":
                mask[token_id] = content_length > 0 and state.type_stack == ["value"]
                continue
            if content_length >= self.max_length:
                continue
            next_stack = apply_token_to_stack(state.type_stack, spec)
            mask[token_id] = next_stack is not None
        return mask

    def step(self, state: FormulaState, token_id: int) -> FormulaState:
        spec = self.vocab.spec(token_id)
        next_ids = list(state.token_ids)
        next_ids.append(token_id)
        if spec.kind == "sep":
            return FormulaState(token_ids=next_ids, type_stack=list(state.type_stack), done=True)
        next_stack = apply_token_to_stack(state.type_stack, spec)
        if next_stack is None:
            raise ValueError(f"Illegal token for current state: {spec.name}")
        done = len(next_ids) - 1 >= self.max_length
        return FormulaState(token_ids=next_ids, type_stack=next_stack, done=done)
