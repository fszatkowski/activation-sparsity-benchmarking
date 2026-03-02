# Optimized LLaDA Diffusion Sparsity Tracking with Jaccard Indices

import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel
from llada_sampler import generate as llada_generate
import llada_sampler as genctx
import time
import csv
import json
from collections import defaultdict
import gc
import sys
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from accelerate import Accelerator

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Global reference to model instance
_global_model_instance = None


class OptimizedSparsityJaccardHook(nn.Module):
    """Single optimized hook that handles both sparsity tracking and Jaccard calculation"""

    def __init__(self, rule, param, layer_name, role="intermediate", track_jaccard=True):
        super().__init__()
        self.rule = rule
        self.param = param
        self.layer_name = layer_name
        self.role = role  # "input" | "intermediate"
        self.track_jaccard = track_jaccard

        # Sparsity tracking (original functionality)
        self.sparsity_stats = []

        # Jaccard tracking - only store what's needed
        self.initial_mask = None
        self.previous_mask = None

        # Efficient aggregation - track raw counts directly
        self.total_intersections_consecutive = 0
        self.total_unions_consecutive = 0
        self.total_intersections_drift = 0
        self.total_unions_drift = 0
        self.total_tokens = 0

        self.per_step_counts = {
            "consecutive": defaultdict(lambda: {"inter": 0, "union": 0, "tokens": 0, 'sum_j': 0, 'jnt': 0}),
            "drift": defaultdict(lambda: {"inter": 0, "union": 0, "tokens": 0}),
        }
        self.per_step_sparsity = defaultdict(lambda: {
            'total_zeros': 0,
            'total_elements': 0,
            'count': 0  # number of forward calls for this step
        })

        # Lightweight token-aware stats (mean/median approximations)
        self.token_stats = {
            "consecutive": defaultdict(lambda: {"sum_j": 0.0, "cnt": 0}),
            "drift": defaultdict(lambda: {"sum_j": 0.0, "cnt": 0}),
        }

        # Control flags for performance
        self.current_step = -1

        # Mask snapshot storage for stability analysis
        self.enable_mask_snapshots = True  # Set to True in profiling mode
        self.mask_snapshots = {}  # {step: {'indices': list_of_arrays, 'magnitudes': optional}}

        # Magnitude tracking for correlation ===
        self.enable_magnitude_tracking = True
        self.magnitude_at_t0 = True  # [H] array of initial magnitudes
        self.neuron_frequency = True  # [H] counter of how often each neuron is active

    def set_jaccard_enabled(self, enabled):
        """Toggle Jaccard tracking for performance during loglikelihood"""
        self.track_jaccard = enabled

    def should_save_snapshot(self, step):
        """Check if snapshot should be saved at this step"""
        if not self.enable_mask_snapshots or step is None:
            return False

        # Check if step matches ANY of the intervals
        for interval in self.snapshot_every:
            if step % interval == 0:
                return True
        return False

    def pre_forward_hook(self, module, input):
        # input[0] is the tensor fed into this Linear; for ff_out it's "intermediate",
        # for ff_in/up/gate it's the residual input to MLP
        x = input[0]

        # Apply sparsity
        if self.rule == 'topp':
            sparse, mask = self._topp_sparsify_with_mask(x, self.param)
        elif self.rule == 'topk':
            sparse, mask = self._topk_sparsify_with_mask(x, self.param)
        else:
            sparse = x
            mask = torch.ones_like(x, dtype=torch.bool)

        # Sparsity stats
        num_zero = (sparse == 0).sum().item()
        total = sparse.numel()
        sparsity_pct = (num_zero / total) * 100 if total > 0 else 0

        # Append to overall list
        self.sparsity_stats.append(sparsity_pct)

        # Aggregate per diffusion step
        step = getattr(genctx, "CURRENT_DIFFUSION_STEP", None)
        if step is not None:
            s = self.per_step_sparsity[int(step)]
            s['total_zeros'] += num_zero
            s['total_elements'] += total
            s['count'] += 1

        # Jaccard tracking (once per diffusion step)
        step = getattr(genctx, "CURRENT_DIFFUSION_STEP", None)
        if self.track_jaccard and step is not None:
            if step != self.current_step:
                self.current_step = step
                self._update_jaccard_tracking(mask, step)

                # Snapshot storage for stability analysis ===
                if self.should_save_snapshot(step):
                    # Store sparse indices per token
                    snapshot_data = self._extract_sparse_snapshot(mask, x)
                    self.mask_snapshots[int(step)] = snapshot_data

                # Magnitude tracking at t=0 ===
                if self.enable_magnitude_tracking and step is not None:
                    if step == 0 and self.magnitude_at_t0 is None:
                        # Store initial magnitudes (average across batch and seq)
                        self.magnitude_at_t0 = x.abs().mean(dim=tuple(range(x.dim() - 1))).detach().cpu()
                        # Initialize frequency counter
                        H = mask.shape[-1]
                        self.neuron_frequency = torch.zeros(H, dtype=torch.long)

                    # Update frequency counter
                    if self.neuron_frequency is not None:
                        # Flatten mask to [total_tokens, H] and count
                        flat_mask = mask.view(-1, mask.shape[-1])
                        self.neuron_frequency += flat_mask.sum(dim=0).cpu()

        # Return sparsified tensor in place of input[0]
        return (sparse,) + input[1:]

    def _extract_sparse_snapshot(self, mask, x=None):
        """
        Extract sparse representation of mask per token.

        Returns dict with 'indices' (list of arrays) and optional 'magnitudes'
        """
        # Reshape to [total_tokens, H]
        if mask.dim() == 3:  # [B, T, H]
            B, T, H = mask.shape
            flat_mask = mask.view(B * T, H)
        elif mask.dim() == 2:  # [B, H] or [T, H]
            flat_mask = mask
        else:
            flat_mask = mask.view(-1, mask.shape[-1])

        # Extract indices per token (store as list of numpy arrays for efficiency)
        indices_per_token = []
        for i in range(flat_mask.shape[0]):
            active_idx = flat_mask[i].nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int16)
            indices_per_token.append(active_idx)

        snapshot = {'indices': indices_per_token}

        # Optionally store magnitudes at active positions
        if x is not None and self.enable_magnitude_tracking:
            flat_x = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
            mags_per_token = []
            for i in range(flat_mask.shape[0]):
                active_idx = flat_mask[i].nonzero(as_tuple=True)[0]

                # Handle any dtype (BFloat16, Float32, etc.)
                tensor_mags = flat_x[i, active_idx].abs().detach()

                # Convert to float32 if needed (handles BFloat16)
                if tensor_mags.dtype == torch.bfloat16:
                    tensor_mags = tensor_mags.float()

                mags = tensor_mags.cpu().numpy().astype(np.float16)
                mags_per_token.append(mags)

            snapshot['magnitudes'] = mags_per_token

        return snapshot

    def get_mask_snapshots(self):
        """Return all stored mask snapshots"""
        return self.mask_snapshots

    def get_correlation_data(self):
        """Return data for neuron correlation analysis"""
        if not self.enable_magnitude_tracking:
            return None

        # Get final mask (last snapshot or from neuron_frequency)
        if self.mask_snapshots:
            last_step = max(self.mask_snapshots.keys())
            final_indices = self.mask_snapshots[last_step]['indices']
            # Merge all tokens' indices into a set
            final_mask_set = set()
            for idx_array in final_indices:
                final_mask_set.update(idx_array.tolist())
        else:
            final_mask_set = None

        return {
            't0_magnitudes': self.magnitude_at_t0,
            'neuron_frequency': self.neuron_frequency,
            'final_mask_set': final_mask_set
        }

    def get_per_step_sparsity(self):
        """
        Return dict: {step: {'mean_sparsity': float, 'count': int}}
        """
        out = {}
        for step_id, s in self.per_step_sparsity.items():
            if s['total_elements'] > 0:
                mean_sparsity = (s['total_zeros'] / s['total_elements']) * 100
            else:
                mean_sparsity = 0.0
            out[int(step_id)] = {
                'mean_sparsity': mean_sparsity,
                'count': s['count'],
                'total_zeros': s['total_zeros'],
                'total_elements': s['total_elements']
            }
        return out

    def _topp_sparsify_with_mask(self, tensor, p_fraction):
        """Optimized top-p with mask return.
        the mask is built independently for each batch sample i over all its features flattened into one vector.
        If the input tensor is [B, T, H], you reshape to [B, T×H] and select the top‑p features per batch row i; when you reshape back, the mask has shape [B, T, H], so it simultaneously marks active positions across all tokens and all hidden features for that sample. It is not applied only along sequence length; it spans the full flattened feature axis T×H per sample.
        """

        # Handle different shapes efficiently
        if tensor.dim() == 2:
            batch_size, hidden_dim = tensor.shape
            flat = tensor
        else:
            batch_size = tensor.shape[0]
            flat = tensor.view(batch_size, -1)

        masks = []
        sparse_flat = torch.zeros_like(flat)

        # Vectorized processing where possible
        for i in range(batch_size):
            abs_vals = flat[i].abs()
            sorted_vals, sorted_indices = torch.sort(abs_vals, descending=True)

            cum_sum = torch.cumsum(sorted_vals, dim=0)
            if cum_sum[-1] > 0:
                cum_sum = cum_sum / cum_sum[-1]

            keep_mask_sorted = cum_sum <= p_fraction / 100
            if not keep_mask_sorted.any():
                keep_mask_sorted[0] = True

            mask = torch.zeros_like(abs_vals, dtype=torch.bool)
            mask[sorted_indices[keep_mask_sorted]] = True
            masks.append(mask)

            sparse_flat[i][mask] = flat[i][mask]

        mask_tensor = torch.stack(masks).view(tensor.shape)
        sparse_tensor = sparse_flat.view(tensor.shape)

        return sparse_tensor, mask_tensor

    def _topk_sparsify_with_mask(self, tensor, k_percent):
        """Optimized top-k with mask return"""
        if tensor.dim() == 2:
            batch_size, hidden_dim = tensor.shape
            flat = tensor
        else:
            batch_size = tensor.shape[0]
            flat = tensor.view(batch_size, -1)

        k = max(int(flat.shape[-1] * k_percent / 100), 1)
        _, topk_indices = torch.topk(flat.abs(), k, dim=-1)

        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)

        sparse = torch.zeros_like(flat)
        sparse[mask] = flat[mask]

        return sparse.view(tensor.shape), mask.view(tensor.shape)

    def _update_jaccard_tracking(self, current_mask, step):
        """Track totals and per-step counts without storing all masks"""
        # Set initial/previous at first call
        if self.initial_mask is None:
            self.initial_mask = current_mask.clone().detach()
            self.previous_mask = current_mask.clone().detach()
            return

        if step > 0:
            # Consecutive (current vs previous)
            if self.previous_mask is not None:
                self._update_jaccard_aggregation(current_mask, self.previous_mask, 'consecutive', step)
            # Drift (current vs initial)
            self._update_jaccard_aggregation(current_mask, self.initial_mask, 'drift', step)

        # Move window
        self.previous_mask = current_mask.clone().detach()

    @torch.no_grad()
    def _tokenwise_jaccard_mean(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor [N_tokens] with Jaccard values per token.
        For shape [B, T, H] or [T, H] or [B, H].
        """
        if m1.dim() == 3:
            B, T, H = m1.shape
            a = m1.reshape(B*T, H)
            b = m2.reshape(B*T, H)
        elif m1.dim() == 2:
            a, b = m1, m2
        else:
            a = m1.reshape(-1, m1.shape[-1])
            b = m2.reshape(-1, m2.shape[-1])

        inter = torch.sum(a & b, dim=1)
        union = torch.sum(a | b, dim=1).clamp_min_(1)
        return (inter.to(torch.float32) / union.to(torch.float32))  # [N_tokens]

    def _update_jaccard_aggregation(self, mask1, mask2, jaccard_type, step):
        if mask1.shape != mask2.shape:
            return

        inter = torch.logical_and(mask1, mask2).sum().item()
        union = torch.logical_or(mask1, mask2).sum().item()
        toks = mask1.numel()

        if jaccard_type == 'consecutive':
            self.total_intersections_consecutive += inter
            self.total_unions_consecutive += union
            c = self.per_step_counts["consecutive"][int(step)]
            c["inter"] += inter;
            c["union"] += union;
            c["tokens"] += toks
        else:
            self.total_intersections_drift += inter
            self.total_unions_drift += union
            c = self.per_step_counts["drift"][int(step)]
            c["inter"] += inter;
            c["union"] += union;
            c["tokens"] += toks

        self.total_tokens += toks

        # TOKEN-AWARE
        j_tok = self._tokenwise_jaccard_mean(mask1, mask2)  # [N_tokens]
        if jaccard_type == 'consecutive':
            ts = self.token_stats["consecutive"][int(step)]
        else:
            ts = self.token_stats["drift"][int(step)]
        ts["sum_j"] += float(j_tok.sum().item())
        ts["cnt"] += int(j_tok.numel())  # n of tokens

    def get_tokenaware_per_step(self):
        """
        Returns: {'consecutive': {step: {'mean': float, 'count': int, 'hist': [..]}}, 'drift': {...}}
        So this returns aggregated Jaccard consecutive and drift metrics aggregated by all the tokens
        and batches per step. For a single step, we have a single measure.
        """
        out = {"consecutive": {}, "drift": {}}
        for k, v in self.token_stats["consecutive"].items():
            mean = (v["sum_j"]/v["cnt"]) if v["cnt"]>0 else 0.0
            payload = {"mean": mean, "count": v["cnt"]}
            out["consecutive"][int(k)] = payload
        for k, v in self.token_stats["drift"].items():
            mean = (v["sum_j"]/v["cnt"]) if v["cnt"]>0 else 0.0
            payload = {"mean": mean, "count": v["cnt"]}
            out["drift"][int(k)] = payload
        return out


    def get_jaccard_results(self):
        consecutive_jaccard = (
            self.total_intersections_consecutive / self.total_unions_consecutive
            if self.total_unions_consecutive > 0 else 0.0
        )
        drift_jaccard = (
            self.total_intersections_drift / self.total_unions_drift
            if self.total_unions_drift > 0 else 0.0
        )
        return {
            'consecutive_jaccard': consecutive_jaccard,
            'drift_jaccard': drift_jaccard,
            'consecutive_intersections': self.total_intersections_consecutive,
            'consecutive_unions': self.total_unions_consecutive,
            'drift_intersections': self.total_intersections_drift,
            'drift_unions': self.total_unions_drift,
            'total_tokens': self.total_tokens
        }

    def get_per_step_counts(self):
        """
        Return dict:
          {
            'consecutive': { step: {'inter': int, 'union': int, 'tokens': int}, ... },
            'drift':       { step: {'inter': int, 'union': int, 'tokens': int}, ... }
          }
        """
        # Convert defaultdicts to plain dicts for serialization
        return {
            'consecutive': {int(k): dict(v) for k, v in self.per_step_counts['consecutive'].items()},
            'drift': {int(k): dict(v) for k, v in self.per_step_counts['drift'].items()},
        }

    def reset_for_generation(self):
        """Reset per-generation state (not aggregation counters)"""
        # Explicitly delete to free memory immediately
        if self.initial_mask is not None:
            del self.initial_mask
        if self.previous_mask is not None:
            del self.previous_mask

        self.initial_mask = None
        self.previous_mask = None
        self.current_step = -1

        # Clear accumulated data
        if hasattr(self, 'mask_snapshots'):
            self.mask_snapshots.clear()

        gc.collect()
        torch.cuda.empty_cache()

@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
            self,
            model_path='',
            mask_id=126336,
            max_length=4096,
            batch_size=32,
            mc_num=128,
            is_check_greedy=True,
            cfg=0.,
            steps=1024,
            gen_length=1024,
            block_length=1024,
            remasking='low_confidence',
            device="cuda",
            **kwargs,
    ):
        super().__init__()

        # Initialize accelerator
        timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))

        accelerator = accelerate.Accelerator(kwargs_handlers=[timeout_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        self.rank_int = self.accelerator.process_index

        # Model setup
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                               **model_kwargs)
        self.model.eval()

        # Device setup
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        # LLaDA parameters
        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking

        # Sparsity parameters
        self.sparsity_rule = kwargs.get("rule", "topp")
        self.sparsity_param = float(kwargs.get("param", 80.0))
        self.script_dir = str(kwargs.get("script_dir", ""))

        # Register optimized hooks

        # Two patterns for: "intermediate" (ff_out) and "input" (ff_in/ff_up/up_proj/gate_proj)
        pat_intermediate = re.compile(r"model\.transformer\.blocks\.[0-9]+\.ff_out$")

        self.hooks = []

        for name, module in self.model.named_modules():
            if pat_intermediate.search(name) and isinstance(module, nn.Linear):
                hook_mid = OptimizedSparsityJaccardHook(
                    self.sparsity_rule, self.sparsity_param, layer_name=f"module.{name}", role="intermediate"
                )
                handle_mid = module.register_forward_pre_hook(hook_mid.pre_forward_hook)
                self.hooks.append((hook_mid, handle_mid))

        print(
            f"Registered {len(self.hooks)} OptimizedSparsityJaccardHooks with {self.sparsity_rule}, param={self.sparsity_param}")

        # Global reference
        global _global_model_instance
        _global_model_instance = self

        # Aggregation storage for final results
        self.jaccard_accumulator = defaultdict(lambda: {
            'consecutive_intersections': 0,
            'consecutive_unions': 0,
            'drift_intersections': 0,
            'drift_unions': 0,
            'total_tokens': 0
        })

        self.jaccard_per_step_acc = defaultdict(lambda: defaultdict(lambda: {
            'consec_inter': 0, 'consec_union': 0,
            'drift_inter': 0, 'drift_union': 0,
            'tokens': 0
        }))  # layer -> step -> counts

        # Snapshot and correlation storage ===
        self.mask_snapshots_storage = defaultdict(dict)  # layer -> step -> snapshot_data
        self.correlation_storage = {}  # layer -> correlation_data

        self.snapshot_every = [32]
        self.enable_profiling_mode(snapshot_every=self.snapshot_every, track_magnitudes=False)

    def enable_profiling_mode(self, snapshot_every=[4, 8, 16, 32], track_magnitudes=False):
        """
        Enable detailed profiling for stability and correlation analysis

        Args:
            snapshot_every: List of intervals at which to save snapshots
                           e.g., [4, 8, 16, 32] saves at steps 0, 4, 8, 12, 16, ...
            track_magnitudes: Whether to track magnitude data
        """
        # Ensure snapshot_every is a list
        if isinstance(snapshot_every, (int, float)):
            snapshot_every = [int(snapshot_every)]
        else:
            snapshot_every = [int(x) for x in snapshot_every]

        for hook, _ in self.hooks:
            hook.enable_mask_snapshots = True
            hook.snapshot_every = snapshot_every
            hook.enable_magnitude_tracking = track_magnitudes

        print(f"Profiling mode enabled: snapshots at intervals {snapshot_every}, magnitudes={track_magnitudes}")

    def disable_profiling_mode(self):
        """Disable profiling to reduce overhead"""
        for hook, _ in self.hooks:
            hook.enable_mask_snapshots = False
            hook.enable_magnitude_tracking = False
        print("Profiling mode disabled")

    def set_jaccard_enabled(self, enabled):
        """Enable/disable Jaccard tracking for performance"""
        for hook, _ in self.hooks:
            hook.set_jaccard_enabled(enabled)

    def reset_generation_state(self):
        """Reset state for new generation"""
        for hook, _ in self.hooks:
            hook.reset_for_generation()

    def accumulate_jaccard_results(self):
        """Accumulate results from hooks into global storage (totals and per-step)."""
        for hook, _ in self.hooks:
            layer_name = hook.layer_name
            results = hook.get_jaccard_results()

            # Totals
            self.jaccard_accumulator[layer_name]['consecutive_intersections'] += results['consecutive_intersections']
            self.jaccard_accumulator[layer_name]['consecutive_unions'] += results['consecutive_unions']
            self.jaccard_accumulator[layer_name]['drift_intersections'] += results['drift_intersections']
            self.jaccard_accumulator[layer_name]['drift_unions'] += results['drift_unions']
            self.jaccard_accumulator[layer_name]['total_tokens'] += results['total_tokens']

            # Per-step
            per_step = hook.get_per_step_counts()
            for step_id, c in per_step['consecutive'].items():
                s = self.jaccard_per_step_acc[layer_name][int(step_id)]
                s['consec_inter'] += int(c['inter'])
                s['consec_union'] += int(c['union'])
                s['tokens'] += int(c['tokens'])
            for step_id, c in per_step['drift'].items():
                s = self.jaccard_per_step_acc[layer_name][int(step_id)]
                s['drift_inter'] += int(c['inter'])
                s['drift_union'] += int(c['union'])
                s['tokens'] += int(c['tokens'])

            # Token-aware per-step mean Jaccard
            tok_per_step = hook.get_tokenaware_per_step()  # {'consecutive': {step: {'mean': float, 'count': int, 'hist': [..]}}, 'drift': {...}}
            # The below code aggregated the jaccard metrics per all diffusion steps. So at the end we have a single measure for jaccard.
            if not hasattr(self, "tokenaware_per_step_acc"):
                self.tokenaware_per_step_acc = defaultdict(lambda: defaultdict(lambda: {
                    'consec_mean_sum': 0.0, 'consec_cnt': 0,
                    'drift_mean_sum': 0.0, 'drift_cnt': 0
                }))
            for step_id, p in tok_per_step['consecutive'].items():
                s = self.tokenaware_per_step_acc[layer_name][int(step_id)]
                s['consec_mean_sum'] += float(p['mean'] * max(p['count'], 1))
                s['consec_cnt'] += int(p['count'])
            for step_id, p in tok_per_step['drift'].items():
                s = self.tokenaware_per_step_acc[layer_name][int(step_id)]
                s['drift_mean_sum'] += float(p['mean'] * max(p['count'], 1))
                s['drift_cnt'] += int(p['count'])

            # Per-step sparsity
            per_step_sparse = hook.get_per_step_sparsity()
            if not hasattr(self, "sparsity_per_step_acc"):
                self.sparsity_per_step_acc = defaultdict(lambda: defaultdict(lambda: {
                    'total_zeros': 0,
                    'total_elements': 0,
                    'count': 0
                }))
            for step_id, sp in per_step_sparse.items():
                s = self.sparsity_per_step_acc[layer_name][int(step_id)]
                s['total_zeros'] += sp['total_zeros']
                s['total_elements'] += sp['total_elements']
                s['count'] += sp['count']

            # THIS PART aggregates mask snapshots:
            if hook.enable_mask_snapshots:
                snapshots = hook.get_mask_snapshots()  # Get from individual hook
                for step_id, snap_data in snapshots.items():
                    # Initialize if first time seeing this layer/step
                    if step_id not in self.mask_snapshots_storage[layer_name]:
                        self.mask_snapshots_storage[layer_name][step_id] = {
                            'indices': [],
                            'magnitudes': []
                        }

                    # APPEND data from this hook/prompt
                    self.mask_snapshots_storage[layer_name][step_id]['indices'].extend(
                        snap_data['indices']
                    )
                    if 'magnitudes' in snap_data:
                        self.mask_snapshots_storage[layer_name][step_id]['magnitudes'].extend(
                            snap_data['magnitudes']
                        )

            # Aggregate correlation data ===
            if hook.enable_magnitude_tracking:
                corr_data = hook.get_correlation_data()
                if corr_data is not None:
                    if layer_name not in self.correlation_storage:
                        self.correlation_storage[layer_name] = {
                            't0_mags': [],
                            'frequencies': [],
                            'final_masks': []
                        }
                    self.correlation_storage[layer_name]['t0_mags'].append(corr_data['t0_magnitudes'])
                    self.correlation_storage[layer_name]['frequencies'].append(corr_data['neuron_frequency'])
                    if corr_data['final_mask_set']:
                        self.correlation_storage[layer_name]['final_masks'].append(corr_data['final_mask_set'])

    def sync_distributed_metrics(self):
        """
        All-reduce metric accumulators across processes so that rank 0
        has globally aggregated counts. Other ranks do nothing afterwards.
        """
        if self.accelerator is None or self.world_size == 1:
            return

        accel = self.accelerator

        # --------- 1) Global Jaccard totals per layer ---------
        for layer_name, data in self.jaccard_accumulator.items():
            buf = torch.tensor([
                data['consecutive_intersections'],
                data['consecutive_unions'],
                data['drift_intersections'],
                data['drift_unions'],
                data['total_tokens'],
            ], device=self.device, dtype=torch.long)

            accel.reduce(buf, reduction="sum")

            # After reduce, every rank has the summed values in buf
            data['consecutive_intersections'] = int(buf[0].item())
            data['consecutive_unions']       = int(buf[1].item())
            data['drift_intersections']      = int(buf[2].item())
            data['drift_unions']             = int(buf[3].item())
            data['total_tokens']             = int(buf[4].item())

        # --------- 2) Per-step Jaccard counts ---------
        for layer_name, step_dict in self.jaccard_per_step_acc.items():
            for step_id, c in step_dict.items():
                buf = torch.tensor([
                    c['consec_inter'],
                    c['consec_union'],
                    c['drift_inter'],
                    c['drift_union'],
                    c['tokens'],
                ], device=self.device, dtype=torch.long)

                accel.reduce(buf, reduction="sum")

                c['consec_inter'] = int(buf[0].item())
                c['consec_union'] = int(buf[1].item())
                c['drift_inter']  = int(buf[2].item())
                c['drift_union']  = int(buf[3].item())
                c['tokens']       = int(buf[4].item())

        # --------- 3) Token-aware per-step Jaccard ---------
        if hasattr(self, "tokenaware_per_step_acc"):
            for layer_name, step_dict in self.tokenaware_per_step_acc.items():
                for step_id, c in step_dict.items():
                    buf = torch.tensor([
                        c['consec_mean_sum'],
                        c['consec_cnt'],
                        c['drift_mean_sum'],
                        c['drift_cnt'],
                    ], device=self.device, dtype=torch.float64)  # float64 safe for sums

                    accel.reduce(buf, reduction="sum")

                    c['consec_mean_sum'] = float(buf[0].item())
                    c['consec_cnt']      = int(buf[1].item())
                    c['drift_mean_sum']  = float(buf[2].item())
                    c['drift_cnt']       = int(buf[3].item())

        # --------- 4) Per-step sparsity counts ---------
        if hasattr(self, "sparsity_per_step_acc"):
            for layer_name, step_dict in self.sparsity_per_step_acc.items():
                for step_id, s in step_dict.items():
                    buf = torch.tensor([
                        s['total_zeros'],
                        s['total_elements'],
                        s['count'],
                    ], device=self.device, dtype=torch.long)

                    accel.reduce(buf, reduction="sum")

                    s['total_zeros']    = int(buf[0].item())
                    s['total_elements'] = int(buf[1].item())
                    s['count']          = int(buf[2].item())

    def _aggregate_by_block(self):
        """
        Build block-level aggregates from per-module jaccard_accumulator, etc.
        Returns a dict: {block_id: {...counts...}} suitable for writing to JSON/CSV.
        """
        block_agg = defaultdict(lambda: {
            "consec_inter": 0,
            "consec_union": 0,
            "drift_inter": 0,
            "drift_union": 0,
            "tokens": 0,
        })

        block_re = re.compile(r"blocks\.(\d+)\.")  # matches block index in layer_name

        for layer_name, data in self.jaccard_accumulator.items():
            m = block_re.search(layer_name)
            if m is None:
                continue
            block_id = int(m.group(1))
            acc = block_agg[block_id]

            acc["consec_inter"] += data["consecutive_intersections"]
            acc["consec_union"] += data["consecutive_unions"]
            acc["drift_inter"] += data["drift_intersections"]
            acc["drift_union"] += data["drift_unions"]
            acc["tokens"] += data["total_tokens"]

        # Convert to final metrics
        block_metrics = {}
        for block_id, acc in block_agg.items():
            cj = acc["consec_inter"] / acc["consec_union"] if acc["consec_union"] > 0 else 0.0
            dj = acc["drift_inter"] / acc["drift_union"] if acc["drift_union"] > 0 else 0.0
            block_metrics[block_id] = {
                "consecutive_jaccard": cj,
                "drift_jaccard": dj,
                "consecutive_intersections": acc["consec_inter"],
                "consecutive_unions": acc["consec_union"],
                "drift_intersections": acc["drift_inter"],
                "drift_unions": acc["drift_union"],
                "total_tokens": acc["tokens"],
            }

        return block_metrics

    def save_final_results(self, filename_base=None):
        """Save aggregated results to files"""
        if filename_base is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Include snapshot intervals in filename
            filename_base = f"{timestamp}_rank{self.rank_int}"
            intervals_str = "_".join(map(str, self.hooks[0][0].snapshot_every)) if self.hooks else "none"
            filename_base = f"{self.script_dir}/{self.sparsity_rule}_{self.sparsity_param}_{intervals_str}_{filename_base}"

        # Compute final Jaccard indices from accumulated counts
        final_results = {
            'layers': {},
            'aggregated_stats': {
                'total_consecutive_jaccard': 0,
                'total_drift_jaccard': 0,
                'total_tokens': 0,
                'num_layers': len(self.jaccard_accumulator)
            },
            'parameters': {
                'sparsity_rule': self.sparsity_rule,
                'sparsity_param': self.sparsity_param,
                'diffusion_steps': self.steps
            }
        }

        total_consecutive_weighted = 0
        total_drift_weighted = 0
        total_tokens_all = 0

        for layer_name, data in self.jaccard_accumulator.items():
            # Calculate layer-wise Jaccard indices
            consecutive_jaccard = (
                data['consecutive_intersections'] / data['consecutive_unions']
                if data['consecutive_unions'] > 0 else 0.0
            )
            drift_jaccard = (
                data['drift_intersections'] / data['drift_unions']
                if data['drift_unions'] > 0 else 0.0
            )

            final_results['layers'][layer_name] = {
                'consecutive_jaccard': consecutive_jaccard,
                'drift_jaccard': drift_jaccard,
                'total_tokens': data['total_tokens'],
                'consecutive_intersections': data['consecutive_intersections'],
                'consecutive_unions': data['consecutive_unions'],
                'drift_intersections': data['drift_intersections'],
                'drift_unions': data['drift_unions']
            }

            # Weighted aggregation
            weight = data['total_tokens']
            total_consecutive_weighted += consecutive_jaccard * weight
            total_drift_weighted += drift_jaccard * weight
            total_tokens_all += weight

        # Overall aggregated statistics
        if total_tokens_all > 0:
            final_results['aggregated_stats'][
                'total_consecutive_jaccard'] = total_consecutive_weighted / total_tokens_all
            final_results['aggregated_stats']['total_drift_jaccard'] = total_drift_weighted / total_tokens_all
            final_results['aggregated_stats']['total_tokens'] = total_tokens_all

        block_metrics = self._aggregate_by_block()
        final_results["layers_by_block"] = {
            str(bid): vals for bid, vals in sorted(block_metrics.items())
        }

        # Save detailed JSON
        json_file = f"{filename_base}.json"
        with open(json_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        # Save summary CSV
        csv_file = f"{filename_base}_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Headers
            writer.writerow(['Layer', 'Consecutive_Jaccard', 'Drift_Jaccard', 'Total_Tokens',
                             'Consec_Intersections', 'Consec_Unions', 'Drift_Intersections', 'Drift_Unions'])

            # Layer data
            for layer_name, data in final_results['layers'].items():
                writer.writerow([
                    layer_name,
                    f"{data['consecutive_jaccard']:.6f}",
                    f"{data['drift_jaccard']:.6f}",
                    data['total_tokens'],
                    data['consecutive_intersections'],
                    data['consecutive_unions'],
                    data['drift_intersections'],
                    data['drift_unions']
                ])

            # Aggregated results
            writer.writerow([])
            writer.writerow(['OVERALL_AGGREGATED',
                             f"{final_results['aggregated_stats']['total_consecutive_jaccard']:.6f}",
                             f"{final_results['aggregated_stats']['total_drift_jaccard']:.6f}",
                             final_results['aggregated_stats']['total_tokens'],
                             '-', '-', '-', '-'])

        print(f"Final results saved to {json_file} and {csv_file}")
        # write per-step JSON and CSV
        per_step_json = f"{filename_base}_per_step.json"
        per_step_csv = f"{filename_base}_per_step.csv"

        # Build per-step structure
        per_step_payload = {
            'layers': {},
            'parameters': final_results['parameters'],
        }

        for layer_name, step_dict in self.jaccard_per_step_acc.items():
            per_step_payload['layers'][layer_name] = {}
            for step_id, c in sorted(step_dict.items(), key=lambda kv: kv[0]):
                consec_j = (c['consec_inter'] / c['consec_union']) if c['consec_union'] > 0 else 0.0
                drift_j = (c['drift_inter'] / c['drift_union']) if c['drift_union'] > 0 else 0.0
                per_step_payload['layers'][layer_name][int(step_id)] = {
                    'consecutive_jaccard': consec_j,
                    'drift_jaccard': drift_j,
                    'consec_intersections': c['consec_inter'],
                    'consec_unions': c['consec_union'],
                    'drift_intersections': c['drift_inter'],
                    'drift_unions': c['drift_union'],
                    'total_tokens': c['tokens'],
                }

        with open(per_step_json, 'w') as f:
            json.dump(per_step_payload, f, indent=2)

        # CSV: Layer, Step, Consecutive_Jaccard, Drift_Jaccard, Tokens, Consec_Intersections, Consec_Unions, Drift_Intersections, Drift_Unions
        with open(per_step_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer', 'Step', 'Consecutive_Jaccard', 'Drift_Jaccard', 'Total_Tokens',
                             'Consec_Intersections', 'Consec_Unions', 'Drift_Intersections', 'Drift_Unions'])
            for layer_name, step_dict in self.jaccard_per_step_acc.items():
                for step_id, c in sorted(step_dict.items(), key=lambda kv: kv[0]):
                    consec_j = (c['consec_inter'] / c['consec_union']) if c['consec_union'] > 0 else 0.0
                    drift_j = (c['drift_inter'] / c['drift_union']) if c['drift_union'] > 0 else 0.0
                    writer.writerow([
                        layer_name, int(step_id),
                        f"{consec_j:.6f}", f"{drift_j:.6f}",
                        c['tokens'], c['consec_inter'], c['consec_union'], c['drift_inter'], c['drift_union']
                    ])

        print(f"Per-step results saved to {per_step_json} and {per_step_csv}")

        # Token-aware per-step export
        if hasattr(self, "tokenaware_per_step_acc"):
            ta_json = f"{filename_base}_tokenaware_per_step.json"
            ta_csv  = f"{filename_base}_tokenaware_per_step.csv"
            payload = {"layers": {}, "parameters": final_results['parameters']}
            for layer_name, step_dict in self.tokenaware_per_step_acc.items():
                payload["layers"][layer_name] = {}
                for step_id, c in sorted(step_dict.items(), key=lambda kv: kv[0]):
                    consec_mean = (c['consec_mean_sum']/c['consec_cnt']) if c['consec_cnt']>0 else 0.0
                    drift_mean  = (c['drift_mean_sum']/c['drift_cnt'])   if c['drift_cnt']>0 else 0.0
                    payload["layers"][layer_name][int(step_id)] = {
                        "consecutive_mean": consec_mean,
                        "consecutive_cnt":  c['consec_cnt'],
                        "drift_mean":       drift_mean,
                        "drift_cnt":        c['drift_cnt'],
                    }
            with open(ta_json, 'w') as f:
                json.dump(payload, f, indent=2)

            with open(ta_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Layer','Step','Consecutive_Mean_TokenJ','Consecutive_Count',
                            'Drift_Mean_TokenJ','Drift_Count'])
                for layer_name, step_dict in self.tokenaware_per_step_acc.items():
                    for step_id, c in sorted(step_dict.items(), key=lambda kv: kv[0]):
                        consec_mean = (c['consec_mean_sum']/c['consec_cnt']) if c['consec_cnt']>0 else 0.0
                        drift_mean  = (c['drift_mean_sum']/c['drift_cnt'])   if c['drift_cnt']>0 else 0.0
                        w.writerow([layer_name, int(step_id),
                                    f"{consec_mean:.6f}", c['consec_cnt'],
                                    f"{drift_mean:.6f}",  c['drift_cnt']])
            print(f"Token-aware per-step results saved to {ta_json} and {ta_csv}")

        # Sparsity per-step export
        if hasattr(self, "sparsity_per_step_acc"):
            sp_json = f"{filename_base}_sparsity_per_step.json"
            sp_csv = f"{filename_base}_sparsity_per_step.csv"

            payload = {"layers": {}, "parameters": final_results['parameters']}
            for layer_name, step_dict in self.sparsity_per_step_acc.items():
                payload["layers"][layer_name] = {}
                for step_id, s in sorted(step_dict.items(), key=lambda kv: kv[0]):
                    if s['total_elements'] > 0:
                        mean_sparsity = (s['total_zeros'] / s['total_elements']) * 100
                    else:
                        mean_sparsity = 0.0
                    payload["layers"][layer_name][int(step_id)] = {
                        "mean_sparsity": mean_sparsity,
                        "count": s['count'],
                        "total_zeros": s['total_zeros'],
                        "total_elements": s['total_elements']
                    }

            with open(sp_json, 'w') as f:
                json.dump(payload, f, indent=2)

            with open(sp_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Layer', 'Step', 'Mean_Sparsity_%', 'Count',
                            'Total_Zeros', 'Total_Elements'])
                for layer_name, step_dict in self.sparsity_per_step_acc.items():
                    for step_id, s in sorted(step_dict.items(), key=lambda kv: kv[0]):
                        if s['total_elements'] > 0:
                            mean_sparsity = (s['total_zeros'] / s['total_elements']) * 100
                        else:
                            mean_sparsity = 0.0
                        w.writerow([
                            layer_name, int(step_id),
                            f"{mean_sparsity:.2f}",
                            s['count'],
                            s['total_zeros'],
                            s['total_elements']
                        ])

            print(f"Per-step sparsity results saved to {sp_json} and {sp_csv}")

        if self.mask_snapshots_storage:
            snap_file = f"{filename_base}_mask_snapshots.npz"
            snap_dict = {}

            # Store metadata about snapshot intervals
            if self.hooks:
                snap_dict['metadata_snapshot_intervals'] = np.array(
                    self.hooks[0][0].snapshot_every if isinstance(self.hooks[0][0].snapshot_every, list)
                    else [self.hooks[0][0].snapshot_every]
                )
                snap_dict['metadata_sparsity_rule'] = self.sparsity_rule
                snap_dict['metadata_sparsity_param'] = self.sparsity_param

            # Separate keys for indices and magnitudes
            for layer_name, step_dict in self.mask_snapshots_storage.items():
                for step_id, snap_data in step_dict.items():
                    key = f"{layer_name}_step_{step_id}"

                    snap_dict[f"{key}_indices"] = np.array(
                        snap_data['indices'], dtype=object
                    )

                    if snap_data['magnitudes']:
                        snap_dict[f"{key}_magnitudes"] = np.array(
                            snap_data['magnitudes'], dtype=object
                        )

            np.savez_compressed(snap_file, **snap_dict)
            print(f"Mask snapshots saved to {snap_file}")

        # Save correlation analysis ===
        if self.correlation_storage:
            corr_file = f"{filename_base}_neuron_correlation.json"
            corr_payload = {}
            for layer_name, data in self.correlation_storage.items():
                # Average across prompts
                avg_t0_mag = torch.stack(data['t0_mags']).mean(dim=0) if data['t0_mags'] else None
                avg_freq = torch.stack(data['frequencies']).sum(dim=0) if data['frequencies'] else None

                # Compute correlation if we have final masks
                if avg_t0_mag is not None and data['final_masks']:
                    # Merge final masks across prompts
                    all_final = set.union(*data['final_masks'])
                    H = len(avg_t0_mag)
                    final_binary = torch.zeros(H, dtype=torch.bool)
                    final_binary[list(all_final)] = True

                    # Compute correlations
                    import scipy.stats
                    mag_corr = scipy.stats.spearmanr(
                        avg_t0_mag.numpy(),
                        final_binary.float().numpy()
                    )[0]

                    freq_corr = scipy.stats.spearmanr(
                        avg_freq.numpy(),
                        final_binary.float().numpy()
                    )[0] if avg_freq is not None else None

                    corr_payload[layer_name] = {
                        't0_magnitude_correlation': float(mag_corr),
                        'frequency_correlation': float(freq_corr) if freq_corr else None,
                        'final_mask_size': len(all_final),
                        'H': H
                    }

            with open(corr_file, 'w') as f:
                json.dump(corr_payload, f, indent=2)
            print(f"Correlation analysis saved to {corr_file}")

        return json_file, csv_file  # unchanged return; per-step files are extra

    def get_sparsity_statistics(self):
        """Calculate and return comprehensive sparsity statistics across all hooks."""
        if not self.hooks:
            return None
        all_stats = []
        layer_stats = {}
        for hook, _ in self.hooks:
            if hook.sparsity_stats:
                layer_name = hook.layer_name or "unknown_layer"
                layer_stats[layer_name] = {
                    'mean': np.mean(hook.sparsity_stats),
                    'std': np.std(hook.sparsity_stats),
                    'min': np.min(hook.sparsity_stats),
                    'max': np.max(hook.sparsity_stats),
                    'count': len(hook.sparsity_stats)
                }
                all_stats.extend(hook.sparsity_stats)
        if not all_stats:
            return None
        return {
            'overall': {
                'mean_sparsity': np.mean(all_stats),
                'std_sparsity': np.std(all_stats),
                'min_sparsity': np.min(all_stats),
                'max_sparsity': np.max(all_stats),
                'median_sparsity': np.median(all_stats),
                'total_forward_passes': len(all_stats),
                'num_hooks': len(self.hooks)
            },
            'per_layer': layer_stats,
            'sparsity_rule': self.sparsity_rule,
            'sparsity_param': self.sparsity_param
        }

    def print_sparsity_statistics(self):
        """Print detailed sparsity statistics."""
        stats = self.get_sparsity_statistics()
        if stats is None:
            print("No sparsity statistics available.")
            return
        print("\n" + "=" * 70)
        print("SPARSITY STATISTICS REPORT")
        print("=" * 70)
        print(f"Sparsity Rule: {stats['sparsity_rule']}")
        print(f"Sparsity Parameter: {stats['sparsity_param']}")
        print(f"Number of hooks: {stats['overall']['num_hooks']}")
        print(f"Total forward passes: {stats['overall']['total_forward_passes']}")
        print("-" * 70)
        print("OVERALL STATISTICS:")
        print(f"  Mean sparsity: {stats['overall']['mean_sparsity']:.2f}%")
        print(f"  Median sparsity: {stats['overall']['median_sparsity']:.2f}%")
        print(f"  Standard deviation: {stats['overall']['std_sparsity']:.2f}%")
        print(f"  Min sparsity: {stats['overall']['min_sparsity']:.2f}%")
        print(f"  Max sparsity: {stats['overall']['max_sparsity']:.2f}%")
        print("-" * 70)
        print("PER-LAYER STATISTICS:")
        for layer_name, layer_stat in stats['per_layer'].items():
            print(f"  {layer_name}:")
            print(f"    Mean: {layer_stat['mean']:.2f}%")
            print(f"    Std:  {layer_stat['std']:.2f}%")
            print(f"    Min:  {layer_stat['min']:.2f}%")
            print(f"    Max:  {layer_stat['max']:.2f}%")
            print(f"    Count: {layer_stat['count']}")
        print("=" * 70)

    def cleanup_hooks(self):
        """Remove all hooks after evaluation."""
        for _, handle in self.hooks:
            handle.remove()
        self.hooks = []

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # LLaDA methods (unchanged)
    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])
        logits = self.model(batch).logits
        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        # Disable Jaccard tracking during loglikelihood for performance
        self.set_jaccard_enabled(False)

        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        # Re-enable Jaccard tracking
        self.set_jaccard_enabled(True)

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False
        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix
        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        assert max(prompt_len) <= 4096
        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
                torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        ### file that will record prompt lengths ###
        if not hasattr(self, "_plog_file"):
            self._plog_file = open(f"run_prompt_lengths_{self.rank_int}.jsonl", "w")
        if not hasattr(self, "_plog_idx"):
            self._plog_idx = 0

        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        # structure of the dataset
        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        # Track original size BEFORE padding
        original_size = len(ds)

        # Dataset padding for multi-GPU
        if self.accelerator is not None:
            world_size = self.accelerator.num_processes
            remainder = original_size % world_size

            if remainder != 0:
                # Need to pad dataset
                padding_needed = world_size - remainder

                if self.accelerator.is_main_process:
                    print(f"\n{'=' * 70}")
                    print(f"DATASET PADDING")
                    print(f"{'=' * 70}")
                    print(f"Original size: {original_size}")
                    print(f"GPUs: {world_size}")
                    print(f"Padding with: {padding_needed} duplicate examples")
                    print(f"New size: {original_size + padding_needed}")
                    print(f"{'=' * 70}\n")

                # Pad by duplicating last examples
                last_example = ds[-1]
                padding_examples = [last_example] * padding_needed

                # Convert to dict format for concatenation
                all_examples = [dict(ds[i]) for i in range(len(ds))] + padding_examples
                ds = Dataset.from_list(all_examples)
                ds = ds.with_format("torch")
            else:
                if self.accelerator.is_main_process:
                    print(f"\n✓ Dataset size {original_size} evenly divisible by {world_size} GPUs\n")

            # Shard dataset across ranks
            ds = ds.shard(
                num_shards=world_size,
                index=self.accelerator.process_index
            )

            # Calculate how many examples THIS rank should actually return
            # (to exclude padded examples)
            examples_per_rank_ideal = original_size // world_size
            has_extra = self.accelerator.process_index < (original_size % world_size)
            num_real_examples = examples_per_rank_ideal + (1 if has_extra else 0)

            # Verify all ranks have same count
            if self.accelerator.is_main_process:
                print(f"✓ Each GPU processes {len(ds)} examples")
                print(f"✓ Returning {num_real_examples} real results (excluding padding)\n")

            # Synchronize before starting
            self.accelerator.wait_for_everyone()
        else:
            num_real_examples = len(ds)

        out = []
        for i, elem in enumerate(tqdm(
                ds,
                desc=f"Generating (Rank {self.rank_int})",
                disable=(self.accelerator is not None and not self.accelerator.is_local_main_process)
        )):
            # Reset generation state for each prompt
            self.reset_generation_state()

            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            # record sequence length for this prompt
            prompt_len = int(prompt.shape[1])
            rec = {
                "seq_idx": int(self._plog_idx),
                "prompt_len": prompt_len,
                "gen_length": int(self.gen_length),
                "block_length": int(self.block_length),
            }
            self._plog_file.write(json.dumps(rec) + "\n")
            self._plog_file.flush()
            self._plog_idx += 1

            try:
                generated_answer = llada_generate(self.model, prompt,
                                                  steps=self.steps,
                                                  gen_length=self.gen_length,
                                                  block_length=self.block_length,
                                                  temperature=0,
                                                  cfg_scale=self.cfg,
                                                  remasking=self.remasking,
                                                  mask_id=self.mask_id)

                generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:],
                                                         skip_special_tokens=False)

                # Apply stop sequences
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                # Remove special tokens
                generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)

                # Ensure non-empty response ===
                if not generated_answer or len(generated_answer.strip()) == 0:
                    print(f"[Rank {self.rank_int}] Warning: Empty response at index {i}, using placeholder")
                    generated_answer = "pass"  # Valid Python code for MBPP/HumanEval
                # ================================================

            except Exception as e:
                print(f"[Rank {self.rank_int}] Error generating at index {i}: {e}")
                generated_answer = "pass"  # Fallback on error

            out.append(generated_answer)

            # Periodic synchronization every 10 examples
            if self.accelerator is not None and i > 0 and i % 10 == 0:
                self.accelerator.wait_for_everyone()

                # Optional periodic memory cleanup every 20 examples
                if i % 20 == 0:
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()

        # Filter out padded results before returning
        if self.accelerator is not None and len(out) > num_real_examples:
            if self.accelerator.is_main_process:
                print(
                    f"\n⚠️  Filtering results: {len(out)} → {num_real_examples} (removing {len(out) - num_real_examples} padded)")
            out = out[:num_real_examples]

        # === FINAL VALIDATION: Ensure all responses are non-empty ===
        for i in range(len(out)):
            if not out[i] or len(out[i].strip()) == 0:
                print(f"[Rank {self.rank_int}] Post-filter warning: Empty response at position {i}")
                out[i] = "pass"
        # =============================================================

        # Log final counts for debugging
        print(f"[Rank {self.rank_int}] Completed {len(out)} generations (expected {num_real_examples})")

        # Accumulate results after all generations
        self.accumulate_jaccard_results()

        # Synchronize after accumulation
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        # Close log file
        if hasattr(self, "_plog_file") and not self._plog_file.closed:
            self._plog_file.close()

        # Final synchronization before return
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":

    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    checks = {
        'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC': '7200',
        'NCCL_ASYNC_ERROR_HANDLING': '1',
        'NCCL_P2P_LEVEL': 'NVL'
    }

    all_ok = True
    for key, expected in checks.items():
        actual = os.environ.get(key, 'NOT SET')
        status = '✓' if actual == expected else '✗'
        print(f"{status} {key:40s} = {actual:20s} (expected: {expected})")
        if actual != expected:
            all_ok = False

    print("=" * 70)
    if all_ok:
        print("✓ All environment variables correctly set!")
    else:
        print("✗ Some environment variables are missing or incorrect!")
        print("Run this before your main script:")
        print()
        for key, expected in checks.items():
            print(f"export {key}={expected}")
        sys.exit(1)

    set_seed(1234)
    print("Starting LLaDA evaluation with optimized sparsity and Jaccard tracking...")

    try:
        cli_evaluate()
        print("\n✓ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n✗ Evaluation failed with error: {e}")
        import traceback

        traceback.print_exc()

        # Try to save partial results even if evaluation failed
        if _global_model_instance is not None:
            print("\nAttempting to save partial results...")
            try:
                if _global_model_instance.accelerator is not None:
                    _global_model_instance.accelerator.wait_for_everyone()
                    _global_model_instance.sync_distributed_metrics()
                    _global_model_instance.accelerator.wait_for_everyone()

                if _global_model_instance.rank == 0:
                    _global_model_instance.save_final_results()
                    print("✓ Partial results saved successfully")
            except Exception as save_error:
                print(f"✗ Could not save partial results: {save_error}")

        sys.exit(1)

    print("\nProcessing final results...")

    if _global_model_instance is not None:
        # If distributed, first synchronize metrics across ranks
        if _global_model_instance.accelerator is not None:
            print("Synchronizing metrics across ranks...")
            _global_model_instance.accelerator.wait_for_everyone()
            _global_model_instance.sync_distributed_metrics()
            _global_model_instance.accelerator.wait_for_everyone()
            print("✓ Synchronization complete")

        # Only rank 0 prints and writes files
        if _global_model_instance.rank == 0:
            print("\nGenerating statistics and saving results...")
            _global_model_instance.print_sparsity_statistics()
            _global_model_instance.save_final_results()
            print("\n✓ All results saved successfully!")

        # All ranks clean up hooks
        _global_model_instance.cleanup_hooks()
        print(f"[Rank {_global_model_instance.rank}] ✓ Hooks cleaned up successfully")
    else:
        print("Warning: No model instance available for results processing.")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)