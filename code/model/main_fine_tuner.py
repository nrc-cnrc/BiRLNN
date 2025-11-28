import sys, os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import json

repo = '/home/jameshko/Documents/birlnn_latest/BIMODAL'
sys.path.insert(0, os.path.join(repo, 'model'))

from sample import Sampler
from rl_reinforce import PPOFineTuner, RLFineTuner, make_reward_fn_list
from helper import clean_molecule
import selfies as sf
from visualize_generated import plot_qed_vs_sas, save_top_molecule_images
from rl_reinforce import sascorer as sa_mod
from retrosynthesis_aizynth import analyze_and_plot_retro_for_run
from seeds import seeds_for_model, get_seed_names

# RDKit for fingerprints to approximate a latent-like diversity metric
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch.nn.functional as F
def compute_log_probs_from_model_strings(sampler, model_seqs: list[str]):
	"""Compute per-position log-probabilities and masks for model-side sequences (SELFIES/SMILES).

	Returns (logp, mask) with shape (B, M) each, where M is sampler._molecular_size.
	"""
	enc = sampler._encoder.encode(model_seqs)
	# Ensure model is in training mode for backprop through LSTM
	sampler._model._lstm.train()
	B = enc.shape[0]
	V = enc.shape[2] if enc.ndim == 3 else sampler._encoding_size
	M = sampler._molecular_size
	pad_idx = getattr(sampler._encoder, 'pad_index', None)

	device = sampler._model._device
	logp = torch.zeros((B, M), dtype=torch.float32, device=device)
	mask = torch.zeros((B, M), dtype=torch.float32, device=device)

	model_type = getattr(sampler, '_model_type', '')

	for i in range(B):
		# Build a full-length token matrix with PAD defaults
		if enc.ndim == 3:
			seq_i = enc[i]  # (L_i, V)
		else:
			seq_i = np.zeros((0, V), dtype=np.float32)

		L_i = seq_i.shape[0]
		final_np = np.zeros((M, V), dtype=np.float32)
		if pad_idx is not None:
			final_np[:, pad_idx] = 1.0
		place = min(M, L_i)
		if place > 0:
			final_np[:place, :] = seq_i[:place, :]

		final = torch.from_numpy(final_np.astype(np.float32)).to(device)

		# Per-sequence accumulators
		logp_seq = torch.zeros((M,), dtype=torch.float32, device=device)
		mask_seq = torch.zeros((M,), dtype=torch.float32, device=device)

		if model_type == 'FBRNN':
			# FBRNN: synchronous two-headed predictions driven by pairs (right_ctx, left_ctx)
			mid = (M - 1) // 2
			sampler._model._lstm.new_sequence(1, device)
			steps = M // 2
			for s in range(steps):
				# Build input pair
				if s == 0:
					r_idx = mid
					l_idx = mid
				else:
					r_idx = mid + s
					l_idx = mid - (s - 1)
				in_pair = torch.cat([final[r_idx, :], final[l_idx, :]], dim=0).view(1, 1, -1)
				fwd, back = sampler._model._lstm(in_pair)
				# Targets for this step
				tgt_f_idx = mid + 1 + s
				tgt_b_idx = mid - s
				if 0 <= tgt_f_idx < M:
					tgt_vec = final[tgt_f_idx]
					if pad_idx is not None:
						is_pad = (tgt_vec.argmax().item() == pad_idx)
					else:
						is_pad = (tgt_vec.sum().item() == 0.0)
					if not is_pad:
						logp_step = F.log_softmax(fwd[-1, 0, :], dim=-1)[tgt_vec.argmax().item()]
						logp_seq[tgt_f_idx] = logp_step
						mask_seq[tgt_f_idx] = 1.0
				if 0 <= tgt_b_idx < M:
					tgt_vec = final[tgt_b_idx]
					if pad_idx is not None:
						is_pad = (tgt_vec.argmax().item() == pad_idx)
					else:
						is_pad = (tgt_vec.sum().item() == 0.0)
					if not is_pad:
						logp_step = F.log_softmax(back[-1, 0, :], dim=-1)[tgt_vec.argmax().item()]
						logp_seq[tgt_b_idx] = logp_step
						mask_seq[tgt_b_idx] = 1.0
		else:
			# Default: BIMODAL-style alternating window expansion
			mid = (M - 1) // 2
			cur_start = mid
			cur_end = mid + 1
			sampler._model._lstm.new_sequence(1, device)
			for j in range(M - 1):
				prefer_right = (j % 2 == 0)
				if prefer_right:
					if cur_end >= M:
						dir = 'left'
					else:
						dir = 'right'
				else:
					if cur_start <= 0:
						dir = 'right'
					else:
						dir = 'left'

				inp = final[cur_start:cur_end, :].unsqueeze(1)
				pred = sampler._model._lstm(inp, dir, device)
				tgt_pos = cur_end if dir == 'right' else (cur_start - 1)
				if tgt_pos < 0 or tgt_pos >= M:
					continue
				tgt_vec = final[tgt_pos]
				if pad_idx is not None:
					is_pad = (tgt_vec.argmax().item() == pad_idx)
				else:
					is_pad = (tgt_vec.sum().item() == 0.0)
				if not is_pad:
					logp_step = F.log_softmax(pred, dim=1)[0, tgt_vec.argmax().item()]
					logp_seq[tgt_pos] = logp_step
					mask_seq[tgt_pos] = 1.0

				if dir == 'right' and cur_end < M:
					cur_end += 1
				elif dir == 'left' and cur_start > 0:
					cur_start -= 1

		# write back per-sequence vectors
		logp[i, :] = logp_seq
		mask[i, :] = mask_seq

	return logp, mask


def smiles_list_to_fps(smiles_list):
	# Use modern RDKit MorganGenerator to avoid deprecation warnings
	gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
	fps = []
	for s in smiles_list:
		try:
			mol = Chem.MolFromSmiles(s)
			if mol is None:
				fps.append(None)
				continue
			fp = gen.GetFingerprint(mol)
			fps.append(fp)
		except Exception:
			fps.append(None)
	return fps


def fps_to_pairwise_distance(fps):
	# Convert to a simple vector per SMILES: average Tanimoto distance to others (as a pseudo-latent diversity)
	n = len(fps)
	vec = np.zeros((n, 1), dtype=np.float32)
	for i in range(n):
		if fps[i] is None:
			continue
		dsum = 0.0
		cnt = 0
		for j in range(n):
			if i == j or fps[j] is None:
				continue
			sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
			dsum += (1.0 - sim)
			cnt += 1
		vec[i, 0] = (dsum / max(1, cnt))
	return torch.from_numpy(vec)


def build_sampler(experiment):
	s = Sampler(experiment, base_path=repo)
	return s


def _sanitize_name(name: str) -> str:
	import re
	if not name:
		return ''
	name = name.strip().replace(' ', '_')
	name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
	return name[:64]  # keep it reasonable


def make_rollout_fn(sampler, fold=2, epoch=9, temperature=0.7, user_seed: str = ''):
	# Build current model once
	stor_dir = os.path.join(repo, 'evaluation')
	model_path = os.path.join(stor_dir, sampler._experiment_name, 'models', f'model_fold_{fold}_epochs_{epoch}')
	sampler._model.build(model_path)

	# Build a fixed reference model (initial checkpoint) for KL regularization
	ref_sampler = build_sampler(sampler._experiment_name)
	ref_sampler._model.build(model_path)

	# Determine representation and seed token
	fmt = 'SELFIES' if sampler._file_name.upper().startswith('SELFIES') else 'SMILES'
	# If user provides a non-empty seed, use it; otherwise default to '[G]' (SELFIES) or 'G' (SMILES)
	seed_str = user_seed if (user_seed is not None and len(user_seed) > 0) else ('[G]' if fmt == 'SELFIES' else 'G')
	try:
		seed_token = sampler._encoder.encode([seed_str])[0]
	except Exception:
		# Fallback to null seed if user's seed is incompatible with the chosen representation
		fallback = '[G]' if fmt == 'SELFIES' else 'G'
		seed_token = sampler._encoder.encode([fallback])[0]

	# Use the top-level helper to compute log-probs

	def rollout_fn(batch_size: int, max_len: int, temp: float = temperature):
		seqs = []
		model_seqs = []
		# Sample using sampler/model, one-by-one to keep it simple
		for _ in range(batch_size):
			arr = sampler._model.sample(seed_token, T=temp)
			decoded = sampler._encoder.decode(arr)[0]
			# Clean padding/tokens then convert to SMILES if SELFIES
			if fmt == 'SELFIES':
				# Use the sampler's actual model type for correct padding/token cleanup
				cleaned = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SELFIES')
				try:
					smi = sf.decoder(cleaned)
				except Exception:
					smi = ''
				seqs.append(smi or '')
				model_seqs.append(cleaned)
			else:
				# SMILES: clean padding/tokens and keep as SMILES
				cleaned = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SMILES')
				seqs.append(cleaned)
				model_seqs.append(cleaned)

		# Compute old log-probs and mask using model-side sequences (behavior)
		old_log_probs, mask = compute_log_probs_from_model_strings(sampler, model_seqs)
		# Compute reference policy log-probs on the same sequences (fixed initial model)
		ref_log_probs, _ = compute_log_probs_from_model_strings(ref_sampler, model_seqs)
		# Detach to prevent gradients through behavior policy
		old_log_probs = old_log_probs.detach()
		mask = mask.detach()
		# Ensure shapes are (batch_size, max_len)
		if old_log_probs.shape[1] != max_len:
			pad_cols = max_len - old_log_probs.shape[1]
			if pad_cols > 0:
				old_log_probs = torch.cat([old_log_probs, torch.zeros((batch_size, pad_cols), dtype=torch.float32)], dim=1)
				mask = torch.cat([mask, torch.zeros((batch_size, pad_cols), dtype=torch.float32)], dim=1)
			else:
				old_log_probs = old_log_probs[:, :max_len]
				mask = mask[:, :max_len]

		# Diversity: use fingerprint-based pairwise distance as a proxy for latent diversity
		fps = smiles_list_to_fps(seqs)
		latent = fps_to_pairwise_distance(fps)

		return {
			'seqs': seqs,               # strings for reward (SMILES)
			'model_seqs': model_seqs,   # strings in model representation (SELFIES/SMILES)
			'old_log_probs': old_log_probs,
			'mask': mask,
			'ref_log_probs': ref_log_probs.detach(),
			'latent': latent,
		}

	return rollout_fn


def make_reinforce_sample_fn(sampler, fold=2, epoch=9, temperature=0.7, user_seed: str = ''):
	"""Build a sample_fn(batch_size, max_len, temp) compatible with RLFineTuner.

	- Samples sequences with the current model policy.
	- Returns reward strings as SMILES (decoded and cleaned).
	- Computes per-token log-probs and mask for the model-side sequences (SELFIES/SMILES) using the same model.
	"""
	# Build model once
	stor_dir = os.path.join(repo, 'evaluation')
	model_path = os.path.join(stor_dir, sampler._experiment_name, 'models', f'model_fold_{fold}_epochs_{epoch}')
	sampler._model.build(model_path)

	# Determine representation and seed token
	fmt = 'SELFIES' if sampler._file_name.upper().startswith('SELFIES') else 'SMILES'
	# If user provides a non-empty seed, use it; otherwise default to '[G]' (SELFIES) or 'G' (SMILES)
	seed_str = user_seed if (user_seed is not None and len(user_seed) > 0) else ('[G]' if fmt == 'SELFIES' else 'G')
	try:
		seed_token = sampler._encoder.encode([seed_str])[0]
	except Exception:
		fallback = '[G]' if fmt == 'SELFIES' else 'G'
		seed_token = sampler._encoder.encode([fallback])[0]

	def sample_fn(batch_size: int, max_len: int, temp: float = temperature):
		seqs = []
		logp_rows = []
		mask_rows = []
		for _ in range(batch_size):
			# Prefer model-native sampling with log-prob capture when available (FBRNN)
			if hasattr(sampler._model, 'sample_with_logp'):
				mol, logp_vec, mask_vec = sampler._model.sample_with_logp(seed_token, T=temp)
				decoded = sampler._encoder.decode(mol)[0]
				if fmt == 'SELFIES':
					cleaned_selfies = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SELFIES')
					try:
						smi = sf.decoder(cleaned_selfies)
					except Exception:
						smi = ''
					seqs.append(smi or '')
				else:
					cleaned_smiles = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SMILES')
					seqs.append(cleaned_smiles)
				logp_rows.append(logp_vec)
				mask_rows.append(mask_vec)
			else:
				# Fallback: sample without per-action logp and estimate logp by teacher-forcing
				arr = sampler._model.sample(seed_token, T=temp)
				decoded = sampler._encoder.decode(arr)[0]
				if fmt == 'SELFIES':
					cleaned_selfies = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SELFIES')
					try:
						smi = sf.decoder(cleaned_selfies)
					except Exception:
						smi = ''
					seqs.append(smi or '')
					model_seq = cleaned_selfies
				else:
					cleaned_smiles = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SMILES')
					seqs.append(cleaned_smiles)
					model_seq = cleaned_smiles
				logp_vec, mask_vec = compute_log_probs_from_model_strings(sampler, [model_seq])
				logp_rows.append(logp_vec[0])
				mask_rows.append(mask_vec[0])

		# Stack per-sample vectors to (B, M)
		logp = torch.stack(logp_rows, dim=0)
		mask = torch.stack(mask_rows, dim=0)
		return seqs, logp, mask

	return sample_fn

def generate_from_checkpoint(ep: int, out_dir: str, args: argparse.Namespace, sampler: Sampler, ckpt_dir: str, algo: str = 'reinforce', user_seed: str = ''):
	# Load checkpoint
	tag = 'reinforce' if algo.lower() == 'reinforce' else 'ppo'
	base = f"{args.experiment}_fold{args.fold}_epoch{args.epoch}_step{ep}_{tag}"
	ckpt_path = os.path.join(ckpt_dir, base)
	try:
		sampler._model.build(ckpt_path)
	except Exception as e:
		print(f"[generate] Failed to load checkpoint for episode {ep}: {e}")
		return
	# Sampling setup
	fmt = 'SELFIES' if sampler._file_name.upper().startswith('SELFIES') else 'SMILES'
	seed_str = user_seed if (user_seed is not None and len(user_seed) > 0) else ('[G]' if fmt == 'SELFIES' else 'G')
	try:
		seed_token = sampler._encoder.encode([seed_str])[0]
	except Exception:
		fallback = '[G]' if fmt == 'SELFIES' else 'G'
		seed_token = sampler._encoder.encode([fallback])[0]
	rows = []
	for _ in range(args.gen_per_episode):
		arr = sampler._model.sample(seed_token, T=args.temp)
		decoded = sampler._encoder.decode(arr)[0]
		if fmt == 'SELFIES':
			model_str = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SELFIES')
			try:
				smi = sf.decoder(model_str)
			except Exception:
				smi = ''
		else:
			model_str = clean_molecule(decoded, getattr(sampler, '_model_type', 'BIMODAL'), fmt='SMILES')
			smi = model_str
		# Compute properties
		qed_score = 0.0
		sa_score = 0.0
		n_heavy = 0
		try:
			mol = Chem.MolFromSmiles(smi) if smi else None
			if mol is not None:
				from rdkit.Chem import QED
				qed_score = float(QED.qed(mol))
				sa_score = float(sa_mod.calculateScore(mol))
				n_heavy = int(mol.GetNumHeavyAtoms())
		except Exception:
			pass
		rows.append({'SELFIES': model_str if fmt == 'SELFIES' else '', 'SMILES': smi, 'QED_Score': qed_score, 'N_heavy': n_heavy, 'SA_Score': sa_score})
	# Save CSV
	csv_path = os.path.join(out_dir, f"generated_molecules_epoch{ep}.csv")
	import pandas as pd
	rows_pd = pd.DataFrame(rows)
	rows_pd.to_csv(csv_path, index=False)
	# Save statistics to a separate file
	stats_path = os.path.join(out_dir, f"generated_molecules_epoch{ep}_stats.txt")
	with open(stats_path, 'w') as f:
		qed_values = rows_pd['QED_Score'].values
		sa_values = rows_pd['SA_Score'].values
		f.write(f"Episode {ep} Generation Statistics:\n")
		f.write(f"Total molecules generated: {len(rows)}\n")
		f.write(f"Average QED Score: {np.mean(qed_values):.4f}\n")
		f.write(f"Average SA Score: {np.mean(sa_values):.4f}\n")
		f.write(f"Max QED Score: {np.max(qed_values):.4f}\n")
		f.write(f"Min SA Score: {np.min(sa_values):.4f}\n")
		f.write(f"QED Score Std Dev: {np.std(qed_values):.4f}\n")
		f.write(f"SA Score Std Dev: {np.std(sa_values):.4f}\n")

	print('[generate] Saved', csv_path)
    

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--algo', choices=['ppo', 'reinforce'], default='reinforce')
	parser.add_argument('--experiment', type=str, default='FBRNN_SELFIES_fixed_1024')
	parser.add_argument('--plot-only', action='store_true', default=False, help='Skip training and load saved training history for plotting')
	parser.add_argument('--skip-generation', action='store_true', default=False)
	parser.add_argument('--steps', type=int, default=200)
	parser.add_argument('--max-episodes', type=int, default=100, help='Maximum number of episodes to plot in trajectory (for REINFORCE).')
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--temp', type=float, default=1.0)
	parser.add_argument('--entropy-coef', type=float, default=1e-4, help='Entropy bonus coefficient (REINFORCE).')
	parser.add_argument('--fold', type=int, default=2)
	parser.add_argument('--epoch', type=int, default=9)
	parser.add_argument('--reward-weights', type=str, default='1,0;1,1', help='Optional list of comma-separated weight pairs for [QED,SAS]. Examples: "1,0" or "1,0;1,1" (use ; or | between pairs). If omitted, runs both (1,1) and (1,0) by default.')
	parser.add_argument('--ckpt-every', type=int, default=20, help='Save model checkpoint every K episodes (REINFORCE).')
	parser.add_argument('--gen-episodes', type=str, default='', help='Comma-separated episode indices to generate from checkpoints, e.g., 0,100,200')
	parser.add_argument('--gen-per-episode', type=int, default=1000, help='How many molecules to sample per listed episode checkpoint')
	parser.add_argument('--run-name', type=str, default='', help='Optional run name suffix for output directories')
	parser.add_argument('--aizynth-config', type=str, default='', help='Path to AiZynthFinder config YAML; if set and gen-episodes provided, runs retrosynthesis analysis.')
	parser.add_argument('--aizynth-limit', type=int, default=0, help='Optional cap on number of molecules per episode to analyze with AiZynthFinder (0 = all).')
	parser.add_argument('--seed', type=str, default='', help="Optional initial string to seed generation. For SELFIES, provide tokens like '[C][C][O]'; for SMILES, 'CCO'. Defaults to '[G]' or 'G' if omitted.")
	parser.add_argument('--seed-name', type=str, default='', help="Optional name for the seed; results will be saved under <algo>/<seed-name>/... when provided.")
	parser.add_argument('--use-all-seeds', action='store_true', default=False, help='If set, run once for each seed returned by seeds_for_model(experiment).')
	args = parser.parse_args()

	# Parse reward-weights which may be a list of comma-separated pairs.
	# If omitted, default to running both (1,1) and (1,0).
	import re
	weight_pairs = []
	if args.reward_weights and args.reward_weights.strip():
		try:
			parts = [s.strip() for s in re.split(r'[;|]+', args.reward_weights) if s.strip()]
			for p in parts:
				vals = [v.strip() for v in p.split(',') if v.strip()]
				if len(vals) == 0:
					continue
				if len(vals) == 1:
					wq = float(vals[0]); ws = 0.0
				else:
					wq = float(vals[0]); ws = float(vals[1])
				weight_pairs.append((wq, ws))
		except Exception as e:
			print('[warning] Failed to parse --reward-weights:', e)
			weight_pairs = []
	# default if none provided or parsing failed
	if not weight_pairs:
		weight_pairs = [(1.0, 1.0), (1.0, 0.0)]

	# If requested, run for all seeds returned by seeds_for_model
	if args.use_all_seeds:
		seeds_list = seeds_for_model(args.experiment)
		names_list = get_seed_names()
		# If names_list shorter than seeds_list, generate numeric labels
		if len(names_list) != len(seeds_list):
			names_list = [f'seed_{i}' for i in range(len(seeds_list))]
		for (wq, ws) in weight_pairs:
			for s, n in zip(seeds_list, names_list):
				print(f"Running for weights=({wq},{ws}) seed-name={n} seed={s}")
				run_experiment(args, [wq, ws], s, n)
		# finished all seeds
		return

	# Otherwise run a single experiment (seed and seed-name from args) for each weight pair
	for (wq, ws) in weight_pairs:
		print(f"Running single-seed experiment for weights=({wq},{ws}) seed-name={args.seed_name} seed={args.seed}")
		run_experiment(args, [wq, ws], args.seed, args.seed_name)

	return


def run_experiment(args, reward_weights, seed, seed_name):
	# Build run directory by algorithm, optional seed-name subfolder, weights and optional suffix
	# Reward functions
	rewards = make_reward_fn_list(['qed', 'sas'])
	wq, ws = reward_weights[0], reward_weights[1]
	weights_tag = f"weights_{wq:g}_{ws:g}"
	run_suffix = ("_" + args.run_name) if args.run_name else ""
	algo_dir = args.algo.lower()

	seed_subdir = _sanitize_name(seed_name)
	algo_path = os.path.join(algo_dir, seed_subdir) if seed_subdir else algo_dir

	run_dir = os.path.join(repo, 'evaluation', 'rl', args.experiment, algo_path, f"{weights_tag}{run_suffix}")
	os.makedirs(run_dir, exist_ok=True)

	# Define training history file path
	history_file = os.path.join(run_dir, f'{args.algo}_training_history.json')

	if args.plot_only:
		# Load saved training history
		if not os.path.exists(history_file):
			raise FileNotFoundError(f"Training history file not found: {history_file}. Run training first without --plot-only.")
		with open(history_file, 'r') as f:
			curves = json.load(f)
		print(f"[plot-only] Loaded training history from {history_file}")
		steps = len(curves['mean_score'])
	else:
		sampler = build_sampler(args.experiment)
		if args.algo == 'reinforce':
			# REINFORCE path
			sample_fn = make_reinforce_sample_fn(sampler, fold=args.fold, epoch=args.epoch, temperature=args.temp, user_seed=seed)
			# default AdamW optimizer on model params
			optim = torch.optim.AdamW(sampler._model._lstm.parameters(), lr=1e-4, weight_decay=1e-2)

			# Checkpoint save function (REINFORCE)
			def make_save_fn(sampler, experiment, fold, epoch):
				def _save(step: int, ckpt_dir: str):
					base = f"{experiment}_fold{fold}_epoch{epoch}_step{step}_reinforce"
					base_path = os.path.join(ckpt_dir, base)
					try:
						sampler._model.save(base_path)
						print('Saved checkpoint to', base_path + '.dat')
					except Exception as e:
						print('Checkpoint save failed:', e)
				return _save

			save_model_fn = make_save_fn(sampler, args.experiment, args.fold, args.epoch)
			rl = RLFineTuner(
				sample_fn=sample_fn,
				reward_fns=rewards,
				reward_weights=reward_weights,
				optimizer=optim,
				save_model_fn=save_model_fn,
				entropy_coef=args.entropy_coef,
			)

			steps = args.steps
			max_len = sampler._molecular_size
			ckpt_dir = os.path.join(run_dir, 'checkpoints')
			os.makedirs(ckpt_dir, exist_ok=True)
			curves = rl.train(
				steps=steps,
				batch_size=args.batch,
				max_len=max_len,
				temperature=args.temp,
				log_interval=1,
				checkpoint_dir=ckpt_dir,
				checkpoint_every=max(1, int(args.ckpt_every)),
				save_initial=True,
			)

		else:
			# PPO path
			rollout_fn = make_rollout_fn(sampler, fold=args.fold, epoch=args.epoch, temperature=args.temp, user_seed=seed)
			def eval_logp_fn(model_side_sequences: list[str]) -> torch.Tensor:
				logp, mask = compute_log_probs_from_model_strings(sampler, model_side_sequences)
				return logp

			def make_save_fn(sampler, experiment, fold, epoch):
				def _save(step: int, ckpt_dir: str):
					base = f"{experiment}_fold{fold}_epoch{epoch}_step{step}_ppo"
					base_path = os.path.join(ckpt_dir, base)
					try:
						sampler._model.save(base_path)
						print('Saved checkpoint to', base_path + '.dat')
					except Exception as e:
						print('Checkpoint save failed:', e)
				return _save

			save_model_fn = make_save_fn(sampler, args.experiment, args.fold, args.epoch)
			ppo = PPOFineTuner(
				rollout_fn=rollout_fn,
				reward_fns=rewards,
				reward_weights=reward_weights,
				optimizer=None,
				eval_logp_fn=eval_logp_fn,
				kl_coef=0.01,
				entropy_coef=0.005,
				diversity_coef=0.0,
				save_model_fn=save_model_fn,
				model_params=sampler._model._lstm.parameters(),
			)

			steps = args.steps
			max_len = sampler._molecular_size
			ckpt_dir = os.path.join(run_dir, 'checkpoints')
			curves = ppo.train(
				steps=steps,
				batch_size=args.batch,
				max_len=max_len,
				temperature=args.temp,
				log_interval=1,
				ppo_epochs=4,
				checkpoint_dir=ckpt_dir,
				checkpoint_every=10,
			)

		# Save training history
		with open(history_file, 'w') as f:
			json.dump(curves, f, indent=2)
		print(f"[training] Saved training history to {history_file}")

	# Plot trajectory (for both training and plot-only modes)
	plt.rcParams.update({'font.size': 14})
	max_episodes = min(steps, args.max_episodes)
	pdf_path = os.path.join(run_dir, f'{args.algo}_trajectory.pdf')
	svg_path = os.path.join(run_dir, f'{args.algo}_trajectory.svg')
	x = np.arange(1, steps + 1)
	mean = np.array(curves['mean_score'])
	std = np.array(curves['std_score'])
	plt.figure(figsize=(6,3))
	plt.plot(x[:max_episodes], mean[:max_episodes], label='Combined score (QED - SAS/9)', color='C1')
	plt.fill_between(x[:max_episodes], mean[:max_episodes] - std[:max_episodes], mean[:max_episodes] + std[:max_episodes], color='C1', alpha=0.2, label='Std dev')
	plt.xlabel('Episode')
	plt.ylabel('Combined score')
	# plt.title('RL Trajectory (REINFORCE)')
	# plt.legend()
	plt.tight_layout()
	plt.savefig(pdf_path)
	plt.savefig(svg_path)
	plt.close()
	print('Saved trajectory to', pdf_path, 'and', svg_path)

	# Parse epochs list (allow generation with or without training)
	if args.gen_episodes:
		try:
			episode_list = [int(x.strip()) for x in args.gen_episodes.split(',') if x.strip() != '']
		except Exception:
			episode_list = []
		if episode_list:
			ckpt_dir = os.path.join(run_dir, 'checkpoints')
			if not os.path.isdir(ckpt_dir):
				print(f"[generate] Checkpoint directory not found: {ckpt_dir}")
			else:
				for ep in episode_list:
					if not args.skip_generation:
						sampler = build_sampler(args.experiment)
						generate_from_checkpoint(ep, run_dir, args, sampler, ckpt_dir, algo=args.algo, user_seed=seed)
						# Save top molecules by high QED and low SA, legacy style
						save_top_molecule_images(run_dir, episode_list, criteria=['QED_Score', 'SA_Score'], ascending=[False, True], n=20)
					# Plot distributions and save top molecule images
					plot_qed_vs_sas(run_dir, episode_list, output_name='distribution')
					
				# Optional retrosynthesis analysis via AiZynthFinder
				if args.aizynth_config:
					try:
						limit = None if (not args.aizynth_limit or args.aizynth_limit <= 0) else int(args.aizynth_limit)
						analyze_and_plot_retro_for_run(run_dir, episode_list, args.aizynth_config, smiles_col='SMILES', max_molecules=limit)
						print('[retro] Saved retrosynthesis analysis to', os.path.join(run_dir, 'retrosynthesis'))
					except Exception as e:
						print('[retro] Failed to run AiZynthFinder analysis:', e)


if __name__ == '__main__':
	main()


