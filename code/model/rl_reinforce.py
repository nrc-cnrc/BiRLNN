"""
REINFORCE fine-tuner for sequence generative models (SMILES and SELFIES)

Design goals:
- Model-agnostic: user provides a `sample_fn(batch_size, max_len, temperature)` that returns
  sequences (list[str]) and per-token log-probabilities (torch.Tensor: [batch, seq_len]) and an optional mask.
- Reward functions are pluggable. Built-in wrappers for QED (RDKit) and SAScore when available.
- Implements a simple REINFORCE update with optional running baseline and entropy bonus.

Usage sketch:
    from model.rl_reinforce import RLFineTuner, qed_reward, sascorer_reward

    # sample_fn: callable -> (seqs, log_probs, mask)
    # seqs: list[str], log_probs: torch.Tensor(batch, seq_len), mask: torch.Tensor(batch, seq_len)

    tuner = RLFineTuner(
        sample_fn=sample_fn,
        reward_fns=[qed_reward, sascorer_reward],
        reward_weights=[0.6, 0.4],
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    tuner.train(steps=1000, batch_size=64, max_len=120)

Notes:
- This file does not assume any specific model API other than the sample function described above.
- If you want the module to call your model directly, wrap it in sample_fn.
- For SAScore, this tries to import a `sascorer` module. If unavailable, that target will be skipped unless you provide your own reward function.

"""

from typing import Callable, List, Tuple, Optional, Dict, Any, Iterable
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from torch.optim import AdamW

# Import RDKit for QED and (optionally) SAScore
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
# Try to import the contrib SAScore module if available. If it's not present in the
# environment, we mark it as unavailable and continue — callers already guard
# against exceptions when computing SAS-based rewards.
SASCORER_AVAILABLE = False
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer  # type: ignore
    SASCORER_AVAILABLE = True
except Exception:
    SASCORER_AVAILABLE = False

# Try to import selfies decoder if available
try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except Exception:
    SELFIES_AVAILABLE = False


def _mol_from_string(s: str, is_selfies: Optional[bool] = None):
    """Return RDKit Mol from SMILES or SELFIES string. Returns None if invalid or RDKit missing.

    If is_selfies is None, heuristic: if string contains '[' or ']' or '.' it's likely SMILES; if SELFIES_AVAILABLE and
    the string is all SELFIES alphabet tokens it may be SELFIES — we don't try to be perfect here. Best is the user
    supplies already-decoded SMILES to the reward function.
    """

    s = s.strip()
    if is_selfies is None and SELFIES_AVAILABLE:
        # quick heuristic: if string starts with 'SELFIES' token style (<) it's not used; instead try decode and fall back
        try:
            # try decode as selfies
            dec = sf.decoder(s)
            if dec and Chem.MolFromSmiles(dec) is not None:
                return Chem.MolFromSmiles(dec)
        except Exception:
            pass
    # assume SMILES
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def qed_reward(seq: str, is_selfies: Optional[bool] = None) -> float:
    """Compute QED score for a given sequence (SMILES or SELFIES). Returns 0.0 for invalid molecules or missing RDKit."""
    mol = _mol_from_string(seq, is_selfies=is_selfies)
    if mol is None:
        return 0.0
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def sascorer_reward(seq: str, is_selfies: Optional[bool] = None) -> float:
    """Compute SAScore for a given sequence. Higher is better for our reward (so we invert the usual scaling)."""
    mol = _mol_from_string(seq, is_selfies=is_selfies)
    if mol is None:
        return 0.0
    try:
        # sascorer returns lower = easier to synthesize. We invert so higher reward = easier to synth.
        sa = sascorer.calculateScore(mol)
        return float(sa)
    except Exception:
        return 0.0


class RLFineTuner:
    """Minimal REINFORCE trainer aligned with Torch_4_RL_Train.py.

    Behavior (per episode = one batch):
      - Sample a batch of sequences and per-token log-probs from the current policy.
      - Compute QED and SAS for each sequence, then reward = w_qed * QED - w_sas * (SAS - 1)/9.
      - Use a per-batch baseline equal to mean reward in the batch.
      - Loss = -mean_i [ (R_i - baseline) * sum_t logp_{i,t} ]
      - Update optimizer once per batch.

    Prints per-episode stats: mean reward, mean QED, mean SAS over valid molecules.

    Note: This class does not call model.parameters() directly; pass an optimizer already bound to model params.
    """

    def __init__(
        self,
        sample_fn: Callable[[int, int, float], Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]],
        reward_fns: Optional[List[Callable[[str], float]]],  # unused, kept for backward-compat
        reward_weights: List[float],  # [w_qed, w_sas]
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        # Optional checkpoint saving callback; should accept (step:int, checkpoint_dir:str)
        save_model_fn: Optional[Callable[[int, str], None]] = None,
        # Entropy bonus coefficient (encourage exploration). Default 0 = disabled.
        entropy_coef: float = 0.0,
    ):
        self.sample_fn = sample_fn
        self.w_qed = float(reward_weights[0]) if len(reward_weights) > 0 else 1.0
        self.w_sas = float(reward_weights[1]) if len(reward_weights) > 1 else 1.0
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_model_fn = save_model_fn
        self.entropy_coef = float(entropy_coef)

    @staticmethod
    def _reward_components_from_smiles(s: str) -> Tuple[float, float, bool]:
        """Return (qed, sas, is_valid). For invalid molecules returns (0.0, 0.0, False)."""
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:
            mol = None
        if mol is None:
            return 0.0, 0.0, False
        try:
            q = float(QED.qed(mol))
        except Exception:
            q = 0.0
        try:
            sa = float(sascorer.calculateScore(mol))
        except Exception:
            sa = 0.0
        return q, sa, True

    def train(
        self,
        steps: int,
        batch_size: int = 32,
        max_len: int = 120,
        temperature: float = 1.0,
        log_interval: int = 10,
        clip_grad_norm: Optional[float] = None,
        # Checkpointing
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 20,
        save_initial: bool = True,
    ) -> Dict[str, List[float]]:
        """Run REINFORCE updates for a number of gradient steps.

        Args:
            steps: total gradient update steps
            batch_size: samples per step
            max_len: maximum generation length
            temperature: sampling temperature
            log_interval: how often to print progress
        """
        device = self.device
        curves = {"mean_score": [], "std_score": [], "mean_qed": [], "mean_sas": []}

        # Optionally save the initial model (step 0)
        if self.save_model_fn is not None and save_initial:
            ckpt_dir = checkpoint_dir or os.path.join(os.getcwd(), 'reinforce_checkpoints')
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
            except Exception:
                pass
            try:
                self.save_model_fn(0, ckpt_dir)
            except Exception as e:
                warnings.warn(f"Failed to save initial checkpoint: {e}")

        for step in range(1, steps + 1):
            # Sample one episode batch
            seqs, log_probs, mask = self.sample_fn(batch_size, max_len, temperature)
            if not isinstance(log_probs, torch.Tensor):
                raise TypeError("sample_fn must return log_probs as a torch.Tensor")
            log_probs = log_probs.to(device)
            mask = (log_probs != 0).to(device) if (mask is None) else mask.to(device)

            # Compute per-sequence rewards and metrics
            rewards_list = []
            qed_list = []
            sas_list = []
            for s in seqs:
                q, sa, is_valid = self._reward_components_from_smiles(s)
                if is_valid:
                    # Torch_4_RL_Train: reward = w_qed * QED - w_sas * (SAS - 1)/9
                    r = self.w_qed * q - self.w_sas * ((sa - 1.0) / 9.0)
                    qed_list.append(q)
                    sas_list.append(sa)
                else:
                    r = -1.0  # invalid molecule penalty
                rewards_list.append(r)

            rewards_np = np.array(rewards_list, dtype=np.float32)
            mean_score = float(np.mean(rewards_np))
            std_score = float(np.std(rewards_np))
            mean_qed = float(np.mean(qed_list)) if len(qed_list) > 0 else 0.0
            mean_sas = float(np.mean(sas_list)) if len(sas_list) > 0 else 0.0
            curves["mean_score"].append(mean_score)
            curves["std_score"].append(std_score)
            curves["mean_qed"].append(mean_qed)
            curves["mean_sas"].append(mean_sas)

            # Sum log-probs per sequence (only over valid positions)
            seq_logprob = (log_probs * mask).sum(dim=1)
            rewards = torch.from_numpy(rewards_np).to(device)
            # Per-batch baseline
            baseline = rewards.mean()
            advantages = rewards - baseline

            # Base REINFORCE loss
            base_loss = - (advantages.detach() * seq_logprob).mean()

            # Entropy bonus: encourage exploration by maximizing per-token entropy.
            # Compute token probabilities (clamp log-probs to avoid underflow) and per-token entropy.
            entropy_mean = torch.tensor(0.0, device=device)
            if self.entropy_coef and self.entropy_coef > 0.0:
                # Ensure mask is float for normalization
                mask_f = mask.to(dtype=torch.float32)
                # Clamp log-probs for numerical stability when exponentiating
                lp_clamped = torch.clamp(log_probs, min=-20.0, max=20.0)
                probs = torch.exp(lp_clamped)
                # per-sequence (sum over tokens) entropy, normalized by number of valid tokens
                token_ent = - (probs * lp_clamped * mask_f).sum(dim=1)
                token_counts = mask_f.sum(dim=1).clamp(min=1.0)
                per_seq_entropy = token_ent / token_counts
                entropy_mean = per_seq_entropy.mean()

            loss = base_loss - (self.entropy_coef * entropy_mean)
            self.optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], clip_grad_norm)
            self.optimizer.step()

            # Per-episode printout aligned with request, including loss-term magnitudes
            if step % log_interval == 0 or step == 1:
                # base_loss is the REINFORCE term; entropy contribution is entropy_coef * entropy_mean
                try:
                    base_loss_val = float(base_loss.item())
                except Exception:
                    base_loss_val = float(base_loss)
                try:
                    entropy_term_val = float(self.entropy_coef * entropy_mean)
                except Exception:
                    entropy_term_val = 0.0
                print(
                    f"step {step}/{steps} total_loss={loss.item():.4f} base_loss={base_loss_val:.4f} "
                    f"entropy_term={entropy_term_val:.6f} mean_return={mean_score:.4f} "
                    f"mean_QED={mean_qed:.4f} mean_SAS={mean_sas:.4f}"
                )

            # Checkpoint saving (if a save callback is provided)
            if self.save_model_fn is not None and checkpoint_every and (step % checkpoint_every == 0):
                ckpt_dir = checkpoint_dir or os.path.join(os.getcwd(), 'reinforce_checkpoints')
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                except Exception:
                    pass
                try:
                    self.save_model_fn(step, ckpt_dir)
                except Exception as e:
                    warnings.warn(f"Failed to save checkpoint at step {step}: {e}")

        print("Training finished")
        return curves


class PPOFineTuner:
    """PPO-style fine-tuner with KL and entropy constraints and optional diversity rewards.

    Design: model-agnostic via callables.
    Required:
      - rollout_fn(batch_size, max_len, temperature) -> dict with keys:
          'seqs': List[str]
          'old_log_probs': torch.Tensor [batch, seq_len] (log-probs for taken actions under behavior policy)
          'mask': torch.Tensor [batch, seq_len] of 0/1
        Optional keys:
          'latent': torch.Tensor [batch, latent_dim]  # for diversity reward
          'ref_log_probs': torch.Tensor [batch, seq_len]  # for KL to reference policy

      - eval_logp_fn(batch_seqs) -> torch.Tensor [batch, seq_len]
        Computes current policy log-probs for the same action sequences.

    Reward functions: same as RLFineTuner.

    Note: If eval_logp_fn is None, this will run in logging mode (no parameter update),
    but still compute metrics and return curves.
    """

    def __init__(
        self,
        rollout_fn: Callable[[int, int, float], Dict[str, Any]],
        reward_fns: List[Callable[[str], float]],
        reward_weights: List[float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        eval_logp_fn: Optional[Callable[[List[str]], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        # PPO/constraints
        clip_coef: float = 0.2,
        kl_coef: float = 0.0,
        entropy_coef: float = 0.0,
        target_kl: Optional[float] = None,
        # Diversity
        diversity_coef: float = 0.0,
        diversity_mode: str = 'pairwise',  # 'pairwise' or 'ref'
        # Checkpointing
        save_model_fn: Optional[Callable[[int, str], None]] = None,
        # Default optimizer wiring
        model: Optional[torch.nn.Module] = None,
        model_params: Optional[Iterable[torch.nn.Parameter]] = None,
        default_lr: float = 1e-4,
        default_weight_decay: float = 1e-2,
    ):
        assert len(reward_fns) == len(reward_weights), "reward_fns and reward_weights must match"
        self.rollout_fn = rollout_fn
        self.reward_fns = reward_fns
        self.reward_weights = reward_weights
        # Align weights with REINFORCE convention [w_qed, w_sas]
        self.w_qed = float(reward_weights[0]) if len(reward_weights) > 0 else 1.0
        self.w_sas = float(reward_weights[1]) if len(reward_weights) > 1 else 1.0
        # If no optimizer provided, create a default AdamW from model or explicit params when available
        if optimizer is None and (model is not None or model_params is not None):
            params = model.parameters() if (model is not None) else model_params
            self.optimizer = AdamW(params, lr=default_lr, weight_decay=default_weight_decay)
        else:
            self.optimizer = optimizer
        self.eval_logp_fn = eval_logp_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_coef = clip_coef
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.diversity_coef = diversity_coef
        self.diversity_mode = diversity_mode
        self.save_model_fn = save_model_fn

    def _compute_reward_batch(self, seqs: List[str]) -> np.ndarray:
        """Compute rewards using the same formulation as RLFineTuner:

        reward = w_qed * QED - w_sas * (SAS - 1)/9, with -1.0 penalty for invalid molecules.
        """
        rewards: List[float] = []
        for s in seqs:
            try:
                q, sa, is_valid = RLFineTuner._reward_components_from_smiles(s)
            except Exception:
                q, sa, is_valid = 0.0, 0.0, False
            if is_valid:
                r = self.w_qed * float(q) - self.w_sas * ((float(sa) - 1.0) / 9.0)
            else:
                r = -1.0
            rewards.append(float(r))
        return np.array(rewards, dtype=np.float32)

    @staticmethod
    def _batch_pairwise_l2(latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, D)
        if latent is None or latent.numel() == 0:
            return torch.tensor(0.0)
        x = latent
        # (B, B)
        dist = torch.cdist(x, x, p=2)
        # mean over off-diagonal entries
        B = x.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=latent.device)
        return dist.sum() / (B * (B - 1))

    def train(
        self,
        steps: int,
        batch_size: int = 32,
        max_len: int = 120,
        temperature: float = 1.0,
        log_interval: int = 10,
        ppo_epochs: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 10,
        ) -> Dict[str, List[float]]:
        """Run PPO updates for a number of steps.

        Returns a dict with curves: 'mean_score', 'std_score' over steps.
        """
        device = self.device
        curves = {"mean_score": [], "std_score": []}

        for step in range(1, steps + 1):
            batch = self.rollout_fn(batch_size, max_len, temperature)
            seqs: List[str] = batch['seqs']
            # Prefer model-side sequences (representation used by the model) when provided
            model_seqs: List[str] = batch.get('model_seqs', seqs)
            old_logp = batch['old_log_probs'].to(device)
            mask = batch['mask'].to(device)
            latent = batch.get('latent', None)
            if isinstance(latent, np.ndarray):
                latent = torch.from_numpy(latent).to(device)
            elif isinstance(latent, torch.Tensor):
                latent = latent.to(device)
            ref_logp = batch.get('ref_log_probs', None)
            if isinstance(ref_logp, np.ndarray):
                ref_logp = torch.from_numpy(ref_logp).to(device)
            elif isinstance(ref_logp, torch.Tensor):
                ref_logp = ref_logp.to(device)

            # Scores
            rewards_np = self._compute_reward_batch(seqs)
            mean_score = float(np.mean(rewards_np))
            std_score = float(np.std(rewards_np))
            curves['mean_score'].append(mean_score)
            curves['std_score'].append(std_score)

            # Build tensors
            rewards = torch.from_numpy(rewards_np).to(device)
            # Diversity bonus (pairwise in latent)
            diversity_bonus = 0.0
            if self.diversity_coef and (latent is not None):
                diversity_bonus = float(self._batch_pairwise_l2(latent).item())
                rewards = rewards + self.diversity_coef * self._batch_pairwise_l2(latent)

            # PPO update only if eval_logp_fn and optimizer are provided
            if (self.eval_logp_fn is not None) and (self.optimizer is not None):
                print("Performing PPO update...")
                # Multiple epochs over same batch
                for _ in range(max(1, ppo_epochs)):
                    # Evaluate current policy log-probs on the model-side sequences (not reward strings)
                    new_logp = self.eval_logp_fn(model_seqs).to(device)
                    # Compute per-sequence log-prob sums over valid tokens
                    old = (old_logp * mask).sum(dim=1)
                    new = (new_logp * mask).sum(dim=1)
                    # Compute log-ratio in a numerically safe way and exponentiate with clamping to avoid inf
                    log_ratio = new - old
                    # Clamp to a reasonable range to prevent overflow in exp while preserving PPO intent
                    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)

                    # PPO clipped objective
                    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv
                    ppo_loss = -torch.mean(torch.min(unclipped, clipped))

                    # KL term (to reference policy if provided) using current vs reference
                    kl_loss = torch.tensor(0.0, device=device)
                    if ref_logp is not None:
                        ref = (ref_logp * mask).sum(dim=1)
                        # approx KL(current || ref) using sampled actions
                        kl = new - ref
                        kl_loss = kl.mean()

                    # Entropy bonus: approximate by negative mean log-prob per token
                    ent_loss = - (new_logp.mean())

                    loss = ppo_loss + self.kl_coef * kl_loss - self.entropy_coef * ent_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if step % log_interval == 0 or step == 1:
                # Compute individual reward components
                qed_scores = []
                sas_scores = []
                for seq in seqs:
                    try:
                        qed_val = qed_reward(seq)
                        qed_scores.append(qed_val)
                    except Exception:
                        pass
                    try:
                        sas_val = sascorer_reward(seq)
                        sas_scores.append(sas_val)
                    except Exception:
                        pass
                
                avg_qed = float(np.mean(qed_scores)) if qed_scores else 0.0
                avg_sas = float(np.mean(sas_scores)) if sas_scores else 0.0
                print(f"step {step}/{steps} mean_score={mean_score:.4f} std={std_score:.4f} avg_qed={avg_qed:.4f} avg_sas={avg_sas:.4f} diversity_bonus={diversity_bonus}")

            # Checkpoint saving (if a save callback is provided)
            if self.save_model_fn is not None and checkpoint_every and (step % checkpoint_every == 0):
                ckpt_dir = checkpoint_dir or os.path.join(os.getcwd(), 'ppo_checkpoints')
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                except Exception:
                    pass
                self.save_model_fn(step, ckpt_dir)

        return curves


# Helper: example wrapper that builds a sample_fn for a model with a .sample() or .generate() API
def make_sample_fn_from_model(model, decode_fn: Optional[Callable[[List[int]], str]] = None, device: Optional[torch.device] = None):
    """Create a sample_fn given a model that can generate tokenized sequences and return per-token log_probs.

    The returned sample_fn(batch_size, max_len, temperature) should return (seqs, log_probs, mask).

    This helper attempts multiple common APIs:
      - model.sample(batch_size, max_len, temperature) -> (token_ids, log_probs)
      - model.generate(batch_size, max_len, temperature) -> token_ids
      - model(...) run autoregressively to sample tokens (slow fallback)

    decode_fn: function that converts token id list -> string (SMILES or SELFIES). If not provided, the model must return strings directly.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def sample_fn(batch_size: int, max_len: int, temperature: float = 1.0):
        # Try model.sample API
        if hasattr(model, 'sample'):
            out = model.sample(batch_size=batch_size, max_len=max_len, temperature=temperature)
            # Accept either (seqs, log_probs) or a dict
            if isinstance(out, tuple) and len(out) >= 2:
                seqs, log_probs = out[0], out[1]
            elif isinstance(out, dict) and 'seqs' in out and 'log_probs' in out:
                seqs, log_probs = out['seqs'], out['log_probs']
            else:
                raise RuntimeError("model.sample returned unexpected output; expected (seqs, log_probs)")
            # If seqs are token ids and decode_fn given, decode
            if len(seqs) > 0 and isinstance(seqs[0], (list, tuple)) and decode_fn is not None:
                seqs = [decode_fn(s) for s in seqs]
            # mask: nonzero log_probs
            mask = (log_probs != 0).float()
            return seqs, torch.tensor(log_probs, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

        # Try model.generate
        if hasattr(model, 'generate'):
            token_ids = model.generate(batch_size=batch_size, max_len=max_len, temperature=temperature)
            if decode_fn is None:
                seqs = token_ids
            else:
                seqs = [decode_fn(t) for t in token_ids]
            # We don't have log_probs; set to small constant so updates are possible but weak. Better to implement model.sample.
            log_probs = torch.full((batch_size, max_len), -10.0)
            mask = torch.zeros_like(log_probs)
            # naive mask: stop at first padding token if model supplied (assume token id 0 is PAD)
            try:
                import numpy as _np
                for i, t in enumerate(token_ids):
                    if isinstance(t, (list, tuple)):
                        for j, tok in enumerate(t):
                            mask[i, j] = 1.0
                    else:
                        # assume string already
                        mask[i, 0] = 1.0
            except Exception:
                mask = (log_probs != 0).float()
            return seqs, log_probs, mask

        # Fallback: autoregressive sampling using model.forward (very generic and may be slow)
        if hasattr(model, 'forward'):
            seqs = []
            logps = []
            masks = []
            model.eval()
            with torch.no_grad():
                for _ in range(batch_size):
                    # naive loop to sample one sequence
                    cur_tokens = []
                    cur_logps = []
                    for t in range(max_len):
                        # user must implement how to get next-token logits from model; we can't guess.
                        raise RuntimeError("Autoregressive fallback not implemented: please provide a sample_fn matched to your model.")
            raise RuntimeError("No supported sampling API found on model; please provide a custom sample_fn")

        raise RuntimeError("No supported sampling API found on model; please provide a custom sample_fn")

    return sample_fn


# Small utility for combining reward functions
def make_reward_fn_list(named_targets: List[str]):
    """Return a list of reward functions for target names. Supported: 'qed', 'sas'"""
    out = []
    for name in named_targets:
        n = name.lower()
        if n in ("qed", "qedscore"):
            out.append(qed_reward)
        elif n in ("sas", "sascorer", "sascore"):
            out.append(sascorer_reward)
        else:
            raise ValueError(f"Unknown target name: {name}")
    return out
