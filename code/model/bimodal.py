"""
Implementation of BIMODAL to generate SMILES
"""

import numpy as np
import torch
import torch.nn as nn
from bidir_lstm import BiDirLSTM

# torch.manual_seed(1)
# np.random.seed(5)


class BIMODAL():

    def __init__(self, molecule_size=7, encoding_dim=55, lr=.01, hidden_units=128, pad_index=None):

        self._molecule_size = molecule_size
        self._input_dim = encoding_dim
        self._output_dim = encoding_dim
        self._layer = 2
        self._hidden_units = hidden_units
        self._pad_index = pad_index

        # Learning rate
        self._lr = lr

        # Build new model
        self._lstm = BiDirLSTM(self._input_dim, self._hidden_units, self._layer)

        # Check availability of GPUs
        self._gpu = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()
            print('GPU available')

        # Adam optimizer
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

        # Cross entropy loss; ignore padding index if provided
        ignore_idx = pad_index if pad_index is not None else -100
        self._loss = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='mean')

    def build(self, name=None):
        """Build new model or load model by name
        :param name:    model name
        """

        if (name is None):
            self._lstm = BiDirLSTM(self._input_dim, self._hidden_units, self._layer)

        else:
            # Attempt to load checkpoint. Prefer loading as full module but
            # fall back to loading weights/state_dict for checkpoints saved
            # with older/newer PyTorch versions.
            ckpt_path = name + '.dat'

            # Try loading as full object first. Some checkpoints store full
            # model objects requiring custom classes to be allow-listed for
            # unpickling (PyTorch >=2.6). Attempt to register BiDirLSTM as a
            # safe global before loading.
            try:
                # Ensure that a top-level module named 'bidir_lstm' is importable
                # during unpickling. Some checkpoints reference this module
                # name even when the file lives inside the `model/` folder.
                import sys
                import importlib.util
                import os as _os

                mod_name = 'bidir_lstm'
                if mod_name not in sys.modules:
                    mod_path = _os.path.join(_os.path.dirname(__file__), 'bidir_lstm.py')
                    try:
                        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        sys.modules[mod_name] = mod
                        _bidir_mod = mod
                    except Exception:
                        # If loading by path fails, attempt regular import
                        try:
                            import bidir_lstm as _bidir_mod
                        except Exception:
                            _bidir_mod = None
                else:
                    _bidir_mod = sys.modules.get(mod_name)

                # Prepare list of safe globals if available
                safe_list = []
                if _bidir_mod is not None:
                    try:
                        safe_list.append(_bidir_mod.BiDirLSTM)
                    except Exception:
                        pass

                # Helper to attempt a torch.load inside safe_globals if possible
                def _safe_torch_load(path, **kwargs):
                    if safe_list and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
                        with torch.serialization.safe_globals(safe_list):
                            return torch.load(path, map_location=self._device, **kwargs)
                    else:
                        # Try to register persistently if available
                        if safe_list and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                            try:
                                torch.serialization.add_safe_globals(safe_list)
                            except Exception:
                                pass
                        return torch.load(path, map_location=self._device, **kwargs)

                # First try to load weights/state dict (preferred)
                try:
                    weights = _safe_torch_load(ckpt_path, weights_only=True)
                    # If this succeeded, fall through to state-dict handling below
                    sd = None
                    if isinstance(weights, dict):
                        if 'state_dict' in weights:
                            sd = weights['state_dict']
                        elif 'model_state_dict' in weights:
                            sd = weights['model_state_dict']
                        else:
                            sd = weights
                    else:
                        sd = weights

                    # Create a fresh model and load state dict (be permissive)
                    model = BiDirLSTM(self._input_dim, self._hidden_units, self._layer)
                    try:
                        model.load_state_dict(sd)
                    except Exception:
                        # Strip 'module.' prefix if present and retry
                        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                        model.load_state_dict(new_sd)
                    self._lstm = model
                except Exception as e_weights:
                    # weights-only load failed; try full-object load as a fallback
                    try:
                        self._lstm = _safe_torch_load(ckpt_path, weights_only=False)
                    except TypeError:
                        # older torch: no weights_only kwarg
                        self._lstm = _safe_torch_load(ckpt_path)
            except Exception as exc_full:
                # If full-module loading fails (common with different PyTorch
                # versions), try loading weights only / state_dict and apply to
                # a freshly created BiDirLSTM instance.
                try:
                    # weights_only argument is available in newer torch versions
                    try:
                        weights = torch.load(ckpt_path, map_location=self._device, weights_only=True)
                    except TypeError:
                        # Older torch versions ignore weights_only kwarg
                        weights = torch.load(ckpt_path, map_location=self._device)

                    # Extract state_dict if wrapped in a dict
                    if isinstance(weights, dict):
                        if 'state_dict' in weights:
                            sd = weights['state_dict']
                        elif 'model_state_dict' in weights:
                            sd = weights['model_state_dict']
                        else:
                            sd = weights
                    else:
                        sd = weights

                    # Create a fresh model and load state dict (be permissive)
                    model = BiDirLSTM(self._input_dim, self._hidden_units, self._layer)
                    # Some state dicts may have 'module.' prefix; try to fix keys
                    try:
                        model.load_state_dict(sd)
                    except RuntimeError:
                        # Strip 'module.' prefix if present
                        new_sd = {}
                        for k, v in sd.items():
                            nk = k.replace('module.', '')
                            new_sd[nk] = v
                        model.load_state_dict(new_sd)

                    self._lstm = model

                except Exception as exc_weights:
                    # If loading weights/state_dict also failed, log a warning
                    # and fall back to a freshly initialized model. This keeps
                    # runtime stable across PyTorch version mismatches where
                    # older checkpoints cannot be unpickled.
                    print(f"Warning: failed to load checkpoint '{ckpt_path}': {exc_full}")
                    print("Falling back to a freshly initialized BiDirLSTM model.")
                    self._lstm = BiDirLSTM(self._input_dim, self._hidden_units, self._layer)

        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

    def train(self, data, label, epochs=1, batch_size=1):
        '''Train the model
        :param  data:   data array (n_samples, molecule_size, encoding_length)
        :param  label:  label array (n_samples, molecule_size)
        :param  epochs: number of epochs for the training
        :param  batch_size: batch size for the training
        :return statistic:  array storing computed losses (epochs, batch size)
        '''

        # Number of samples
        n_samples = data.shape[0]

        # Change axes from (n_samples, molecule_size, encoding_dim) to (molecule_size, n_samples, encoding_dim)
        data = np.swapaxes(data, 0, 1)

        # Create tensor from label
        label = torch.from_numpy(label).to(self._device)

        # Calculate number of batches per epoch
        if (n_samples % batch_size) == 0:
            n_iter = n_samples // batch_size
        else:
            n_iter = n_samples // batch_size + 1

        # To store losses
        statistic = np.zeros((epochs, n_iter))

        # Prepare model for training
        self._lstm.train()

        # Iteration over epochs
        for i in range(epochs):

            # Iteration over batches
            for n in range(n_iter):

                # Set gradient to zero for batch
                self._optimizer.zero_grad()

                # Store losses in list
                losses = []

                # Accumulate per-molecule loss into a differentiable tensor
                molecule_loss = torch.zeros(1, device=self._device)

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Reset model with correct batch size
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                # Current batch
                batch_data = torch.from_numpy(data[:, batch_start:batch_end, :].astype('float32')).to(self._device)

                # Initialize start and end position of sequence read by the model
                # Use a safe middle-left start so expanding left/right never
                # produces an out-of-bounds index for even or odd molecule_size.
                # clamp to >=0 for very small molecule sizes.
                start = max(0, (self._molecule_size-1) // 2)
                end = start + 1

                for j in range(self._molecule_size - 1):
                    self._lstm.new_sequence(batch_end - batch_start, self._device)

                    # Select direction for next prediction
                    if j % 2 == 0:
                        dir = 'right'
                    else:
                        dir = 'left'

                    # Predict next token
                    pred = self._lstm(batch_data[start:end], dir, self._device)

                    # Compute loss and extend sequence read by the model
                    if j % 2 == 0:
                        step_tgt = label[batch_start:batch_end, end]
                        # If all targets are padding, skip computing CE (avoid mean over zero elements)
                        if (self._pad_index is not None) and torch.all(step_tgt == self._pad_index):
                            loss = torch.zeros(1).to(self._device)
                        else:
                            loss = self._loss(pred, step_tgt)
                        end += 1

                    else:
                        step_tgt = label[batch_start:batch_end, start - 1]
                        if (self._pad_index is not None) and torch.all(step_tgt == self._pad_index):
                            loss = torch.zeros(1).to(self._device)
                        else:
                            loss = self._loss(pred, step_tgt)
                        start -= 1

                    # Append loss of current position
                    losses.append(loss.item())

                    # Accumulate into molecule_loss so we call backward once per batch
                    # Ensure zero tensors are part of the autograd graph by using add
                    molecule_loss = torch.add(molecule_loss, loss)

                    # Compute some quick stats for the prediction to help debug numeric issues
                    try:
                        pred_min = float(pred.min().item())
                        pred_max = float(pred.max().item())
                        pred_mean = float(pred.mean().item())
                    except Exception:
                        pred_min = pred_max = pred_mean = None

                    # If loss is NaN or Inf, print context and abort for post-mortem
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Numeric issue detected: NaN/Inf loss at epoch={i} batch={n} step={j} dir={dir}", flush=True)
                        print(f" pred_min={pred_min} pred_max={pred_max} pred_mean={pred_mean}", flush=True)
                        try:
                            tgt = label[batch_start:batch_end, end] if (j % 2 == 0) else label[batch_start:batch_end, start - 1]
                            print(f" targets unique={torch.unique(tgt).cpu().numpy()}", flush=True)
                        except Exception:
                            tgt = None

                        # Detailed diagnostics: shapes, finiteness, and target ranges
                        try:
                            print(f" pred.shape={pred.shape} pred.dtype={pred.dtype}", flush=True)
                            print(f" pred.isfinite_all={torch.isfinite(pred).all().item()} pred.isnan_sum={int(torch.isnan(pred).sum().item())} pred.isinf_sum={int(torch.isinf(pred).sum().item())}", flush=True)
                            if tgt is not None:
                                try:
                                    print(f" labels dtype={tgt.dtype} min={int(tgt.min().item())} max={int(tgt.max().item())}", flush=True)
                                except Exception:
                                    pass
                            print(f" vocab_size={pred.shape[1]} ignore_index={getattr(self._loss, 'ignore_index', None)}", flush=True)
                            # Try computing log_softmax and selecting the target log-probabilities
                            try:
                                logp = torch.nn.functional.log_softmax(pred, dim=1)
                                if tgt is not None:
                                    sel = logp[torch.arange(logp.size(0)), tgt]
                                    print(f" sel_logp_min={float(sel.min().item())} sel_logp_max={float(sel.max().item())} sel_logp_mean={float(sel.mean().item())}", flush=True)
                            except Exception as e_sel:
                                print(f"Selecting log-probs failed: {e_sel}", flush=True)
                        except Exception:
                            pass

                        # Dump a compact summary of batch inputs for failing example
                        try:
                            # batch_data is (molecule_size, batch, encoding_dim)
                            # Print the sums per token position and the one-hot positions for the failing step
                            batch_np = batch_data.cpu().numpy()  # small batch
                            # Summed activation per position (over batch and encoding dim)
                            pos_sums = np.sum(batch_np, axis=(1,2))
                            print(f" batch_pos_sums={pos_sums.tolist()}", flush=True)
                            # For the failing next-token position, print target indices per sample
                            if tgt is not None:
                                print(f" target_tokens={tgt.cpu().numpy().tolist()}", flush=True)
                        except Exception:
                            pass

                        raise RuntimeError("Encountered NaN/Inf loss during training")

                    # (grad accumulation performed by adding into molecule_loss; we'll backward after the loop)

                    # Inspect gradients for NaN/Inf after backward
                    try:
                        total_grad_norm_sq = 0.0
                        for p in self._lstm.parameters():
                            if p.grad is not None:
                                gnorm = p.grad.data.norm(2).item()
                                total_grad_norm_sq += gnorm * gnorm
                        total_grad_norm = total_grad_norm_sq ** 0.5
                        if np.isnan(total_grad_norm) or np.isinf(total_grad_norm):
                            print(f"Numeric issue: NaN/Inf gradient norm at epoch={i} batch={n} step={j} dir={dir} grad_norm={total_grad_norm}", flush=True)
                            raise RuntimeError("Encountered NaN/Inf gradient norm after backward")
                    except Exception:
                        # Best-effort; don't crash on debugging inspection
                        pass
                
                # Backpropagate once per batch if there were prediction steps
                if len(losses) > 0:
                    molecule_loss.backward()

                # Clip gradients to avoid explosion / NaNs
                try:
                    torch.nn.utils.clip_grad_norm_(self._lstm.parameters(), max_norm=5.0)
                except Exception:
                    pass

                # Store statistics: loss per token (middle token not included)
                statistic[i, n] = np.sum(losses) / (self._molecule_size - 1) if len(losses) > 0 else 0.0

                # Print loss information for debugging (epoch, batch, mean loss, per-step losses)
                # print(f"Train(epoch={i}) batch={n} mean_loss={statistic[i,n]:.6f} per_step_losses={losses}", flush=True)

                # Perform optimization step
                self._optimizer.step()

        return statistic

    def validate(self, data, label, batch_size=128):
        ''' Validation of model and compute error
        :param data:    test data (n_samples, molecule_size, encoding_size)
        :param label:   label data (n_samples_molecules_size)
        :param batch_size:  batch size for validation
        :return:            mean loss over test data
        '''

        # Use train mode to get loss consistent with training
        self._lstm.train()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Compute tensor of labels
            label = torch.from_numpy(label).to(self._device)

            # Number of samples
            n_samples = data.shape[0]

            # Change axes from (n_samples, molecule_size, encoding_dim) to (molecule_size , n_samples, encoding_dim)
            data = np.swapaxes(data, 0, 1).astype('float32')

            # Initialize loss for complete validation set
            tot_loss = 0

            # Calculate number of batches per epoch
            if (n_samples % batch_size) == 0:
                n_iter = n_samples // batch_size
            else:
                n_iter = n_samples // batch_size + 1

            for n in range(n_iter):

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Data used in this batch
                batch_data = torch.from_numpy(data[:, batch_start:batch_end, :].astype('float32')).to(self._device)

                # Initialize loss for molecule
                molecule_loss = 0

                # Reset model with correct batch size and device
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                # Use same safe initialization as in train().
                start = max(0, (self._molecule_size-1) // 2)
                end = start + 1

                for j in range(self._molecule_size - 1):
                    self._lstm.new_sequence(batch_end - batch_start, self._device)

                    # Select direction for next prediction
                    if j % 2 == 0:
                        dir = 'right'
                    if j % 2 == 1:
                        dir = 'left'

                    # Predict next token
                    pred = self._lstm(batch_data[start:end], dir, self._device)

                    # Extend reading of the sequence
                    if j % 2 == 0:
                        step_tgt = label[batch_start:batch_end, end]
                        if (self._pad_index is not None) and torch.all(step_tgt == self._pad_index):
                            loss = torch.zeros(1).to(self._device)
                        else:
                            loss = self._loss(pred, step_tgt)
                        end += 1

                    if j % 2 == 1:
                        step_tgt = label[batch_start:batch_end, start - 1]
                        if (self._pad_index is not None) and torch.all(step_tgt == self._pad_index):
                            loss = torch.zeros(1).to(self._device)
                        else:
                            loss = self._loss(pred, step_tgt)
                        start -= 1

                    # Sum loss over molecule
                    molecule_loss += loss.item()

                # Add loss per token to total loss (start token and end token not counted)
                tot_loss += molecule_loss / (self._molecule_size - 1)

            return tot_loss / n_iter

    def sample(self, seed_token, T=1):
        '''Generate new molecule
        :param seed_token:      starting token or sequence of tokens. Either shape (vocab,)
                                for a single one-hot token or (L, vocab) for a multi-token seed.
        :param T:               sampling temperature
        :return molecule:       newly generated molecule (1, molecule_length, encoding_length)
        '''

        # Prepare model
        self._lstm.eval()

        # Normalize seed to shape (L, vocab)
        seed = np.asarray(seed_token)
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != self._output_dim:
            raise ValueError(f"BIMODAL.sample: seed has wrong vocab dim {seed.shape[1]} != {self._output_dim}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        with torch.no_grad():
            # Sequence buffer (molecule_size, 1, vocab)
            seq = np.zeros((self._molecule_size, 1, self._output_dim), dtype=np.float32)

            # Place seed centered around the middle index (for even L, mid is inside the left half)
            mid = (self._molecule_size - 1) // 2
            start = mid - (L - 1) // 2
            end = start + L
            if start < 0:
                shift = -start
                start += shift
                end += shift
            if end > self._molecule_size:
                shift = end - self._molecule_size
                start -= shift
                end -= shift

            seq[start:end, 0, :] = seed

            # Convert to tensor
            seq_t = torch.from_numpy(seq).to(self._device)

            # Set reading window
            cur_start = start
            cur_end = end

            # Alternate directions; if one side is full, grow on the other
            for j in range(self._molecule_size - L):
                self._lstm.new_sequence(1, self._device)

                # Choose direction: even -> right, odd -> left, but skip if at boundary
                prefer_right = (j % 2 == 0)
                if prefer_right:
                    if cur_end >= self._molecule_size:
                        dir = 'left'
                    else:
                        dir = 'right'
                else:
                    if cur_start <= 0:
                        dir = 'right'
                    else:
                        dir = 'left'

                pred = self._lstm(seq_t[cur_start:cur_end], dir, self._device)
                token = self.sample_token(np.squeeze(pred.cpu().detach().numpy()), T)

                if dir == 'right' and cur_end < self._molecule_size:
                    seq_t[cur_end, 0, token] = 1.0
                    cur_end += 1
                elif dir == 'left' and cur_start > 0:
                    seq_t[cur_start - 1, 0, token] = 1.0
                    cur_start -= 1
                else:
                    # Should not hit due to boundary checks; continue to next step
                    continue

        return seq_t.cpu().numpy().reshape(1, self._molecule_size, self._output_dim)

    def sample_with_logp(self, seed_token, T=1.0):
        """Generate a molecule with the BIMODAL sampler and return per-position log-probabilities.

        Strategy mirrors sample(): place the seed centered, then alternate growing right/left
        (falling back to the available side if one side is full). We use BiDirLSTM.forward with
        next_prediction argument to obtain logits for the chosen direction, then record the
        log-probability of the sampled token at the filled position.

        Returns:
          molecule: np.ndarray of shape (1, molecule_size, vocab)
          log_probs: torch.Tensor of shape (molecule_size,) with log p(action) at positions where an action was taken, 0 elsewhere
          mask: torch.Tensor of shape (molecule_size,) with 1.0 at positions where an action was taken, 0 elsewhere
        """
        self._lstm.train()

        seed = np.asarray(seed_token)
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != self._output_dim:
            raise ValueError(f"BIMODAL.sample_with_logp: seed has wrong vocab dim {seed.shape[1]} != {self._output_dim}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        device = self._device
        # Keep sequence as torch tensor on device for LSTM calls
        seq_t = torch.zeros((self._molecule_size, 1, self._output_dim), dtype=torch.float32, device=device)

        # Centered placement of seed
        mid = (self._molecule_size - 1) // 2
        start = mid - (L - 1) // 2
        end = start + L
        if start < 0:
            shift = -start
            start += shift
            end += shift
        if end > self._molecule_size:
            shift = end - self._molecule_size
            start -= shift
            end -= shift
        seq_t[start:end, 0, :] = torch.from_numpy(seed.astype(np.float32)).to(device)

        # Initialize hidden state for batch size 1
        self._lstm.new_sequence(1, device)

        taken_positions = []
        taken_logps = []

        # Alternate directions; if one side is full, use the other
        cur_start = start
        cur_end = end
        for j in range(self._molecule_size - L):
            prefer_right = (j % 2 == 0)
            if prefer_right:
                if cur_end >= self._molecule_size:
                    direction = 'left'
                else:
                    direction = 'right'
            else:
                if cur_start <= 0:
                    direction = 'right'
                else:
                    direction = 'left'

            # Input context window
            # Clone context to avoid autograd in-place versioning issues when we later
            # modify seq_t (filled positions become part of subsequent contexts).
            ctx = seq_t[cur_start:cur_end, :, :].clone()
            logits = self._lstm.forward(ctx, next_prediction=direction, device=device)
            logits1d = logits.view(-1)

            # Sample token (temperature-scaled)
            probs = torch.softmax(logits1d / float(T), dim=0)
            token = torch.multinomial(probs, num_samples=1)[0].item()

            # Target position to fill
            tgt_pos = cur_end if direction == 'right' else (cur_start - 1)
            if tgt_pos < 0 or tgt_pos >= self._molecule_size:
                continue

            # One-hot write into sequence buffer
            onehot = torch.zeros((self._output_dim,), dtype=torch.float32, device=device)
            onehot[token] = 1.0
            seq_t[tgt_pos, 0, :] = onehot

            # Log-prob of taken action
            logp_taken = torch.log_softmax(logits1d / float(T), dim=0)[token]
            taken_positions.append(int(tgt_pos))
            taken_logps.append(logp_taken)

            # Update bounds
            if direction == 'right' and cur_end < self._molecule_size:
                cur_end += 1
            elif direction == 'left' and cur_start > 0:
                cur_start -= 1

        # Build differentiable log_probs vector via scatter
        if len(taken_positions) > 0:
            idx_t = torch.tensor(taken_positions, dtype=torch.long, device=device)
            src = torch.stack(taken_logps, dim=0)
            log_probs = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device).scatter(0, idx_t, src)
            mask = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device).scatter(0, idx_t, torch.ones_like(src))
        else:
            log_probs = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device)
            mask = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device)

        molecule = seq_t.detach().cpu().numpy().reshape(1, self._molecule_size, self._output_dim)
        return molecule, log_probs, mask

    def sample_token(self, out, T=1.0):
        ''' Sample token
        :param out: output values from model
        :param T:   sampling temperature
        :return:    index of predicted token
        '''
        # Explicit conversion to float64 avoiding truncation errors
        out = out.astype('float64')

        # Compute probabilities with specific temperature (numerically stable)
        out_T = out / T
        m = np.max(out_T)
        ex = np.exp(out_T - m)
        s = np.sum(ex)
        if not np.isfinite(s) or s <= 0:
            # fallback to uniform to avoid NaNs
            p = np.ones_like(out_T, dtype=np.float64) / float(len(out_T))
        else:
            p = ex / s

        # Generate new token at random
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def save(self, name='test_model'):
        torch.save(self._lstm, name + '.dat')
