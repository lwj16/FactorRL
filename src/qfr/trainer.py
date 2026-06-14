from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from qfr.config import ExperimentConfig
from qfr.env import FormulaEnv, FormulaState
from qfr.pool import CandidateResult, FactorPool
from qfr.policy import PolicyNetwork
from qfr.rpn import Expression, parse_expression
from qfr.tokens import Vocabulary


@dataclass(slots=True)
class Trajectory:
    state: FormulaState
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    expression: Expression | None
    valid: bool


class QFRTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        vocab: Vocabulary,
        env: FormulaEnv,
        pool: FactorPool,
        policy: PolicyNetwork,
    ) -> None:
        self.config = config
        self.vocab = vocab
        self.env = env
        self.pool = pool
        # resolve device as torch.device and move model
        self.device = torch.device(config.device)
        self.policy = policy.to(self.device)
        # if multiple GPUs are visible, wrap with DataParallel for simple multi-GPU
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.policy = torch.nn.DataParallel(self.policy)
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.training.learning_rate,
        )

    def _masked_distribution(self, token_ids: list[int], mask: np.ndarray) -> Categorical:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        logits = self.policy(input_ids)[0]
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        masked_logits = logits.masked_fill(~mask_tensor, -1e9)
        return Categorical(logits=masked_logits)

    def _rollout(self, greedy: bool) -> Trajectory:
        state = self.env.reset()
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        while not state.done:
            mask = self.env.legal_action_mask(state)
            if not mask.any():
                break
            dist = self._masked_distribution(state.token_ids, mask)
            action = torch.argmax(dist.logits) if greedy else dist.sample()
            if not greedy:
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
            state = self.env.step(state, int(action.item()))
        valid = len(state.token_ids) > 1 and state.token_ids[-1] == self.vocab.sep_id
        expression = None
        if valid:
            try:
                expression = parse_expression(state.token_ids, self.vocab)
            except Exception:
                valid = False
        return Trajectory(
            state=state,
            log_probs=log_probs,
            entropies=entropies,
            expression=expression,
            valid=valid,
        )

    def _evaluate(self, trajectory: Trajectory, step: int) -> CandidateResult:
        if not trajectory.valid or trajectory.expression is None:
            return self.pool.invalid_result()
        return self.pool.evaluate_candidate(trajectory.expression, step)

    def train(self) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
        history: list[dict[str, float | str]] = []
        global_step = 0
        progress = tqdm(range(self.config.training.episodes), desc="QFR")
        for episode in progress:
            batch_losses: list[torch.Tensor] = []
            batch_rewards: list[float] = []
            batch_advantages: list[float] = []
            best_result: CandidateResult | None = None
            for _ in range(self.config.training.batch_size):
                sampled = self._rollout(greedy=False)
                greedy = self._rollout(greedy=True)
                sampled_result = self._evaluate(sampled, global_step)
                greedy_result = self._evaluate(greedy, global_step)
                advantage = sampled_result.stats.reward - greedy_result.stats.reward
                if sampled.log_probs:
                    policy_term = torch.stack(sampled.log_probs).sum()
                    entropy_term = torch.stack(sampled.entropies).sum()
                    loss = -(advantage * policy_term) - self.config.training.entropy_weight * entropy_term
                    batch_losses.append(loss)
                batch_rewards.append(sampled_result.stats.reward)
                batch_advantages.append(advantage)
                candidate = sampled_result if sampled_result.stats.reward >= greedy_result.stats.reward else greedy_result
                if best_result is None or candidate.stats.reward > best_result.stats.reward:
                    best_result = candidate
                global_step += 1
            if batch_losses:
                loss = torch.stack(batch_losses).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.training.grad_clip)
                self.optimizer.step()
                loss_value = float(loss.detach().cpu().item())
            else:
                loss_value = 0.0
            if best_result is not None:
                self.pool.maybe_add(best_result)
            record = {
                "episode": episode,
                "loss": loss_value,
                "mean_reward": float(np.mean(batch_rewards)) if batch_rewards else -1.0,
                "mean_advantage": float(np.mean(batch_advantages)) if batch_advantages else 0.0,
                "pool_size": len(self.pool.entries),
                "best_expression": best_result.rendered if best_result is not None else "<none>",
                "best_reward": best_result.stats.reward if best_result is not None else -1.0,
                "best_ic": best_result.stats.mean_ic if best_result is not None else -1.0,
                "best_ir": best_result.stats.ir if best_result is not None else -1.0,
            }
            history.append(record)
            if (episode + 1) % self.config.training.log_every == 0:
                progress.set_postfix(
                    reward=f"{record['mean_reward']:.4f}",
                    pool=record["pool_size"],
                )
        return history, self.pool.summary()
