"""
Implementation of synchronous Forward Backward Models
"""

import numpy as np
import torch
import torch.nn as nn
from two_out_lstm_v2 import TwoOutLSTM_v2

# torch.manual_seed(1)
# np.random.seed(5)


class FBRNN():

    def __init__(self, molecule_size=7, encoding_dim=55, lr=.01, hidden_units=256, pad_index=None):

        self._molecule_size = molecule_size
        self._input_dim = 2 * encoding_dim
        self._layer = 2
        self._hidden_units = hidden_units
        self._pad_index = pad_index

        # Learning rate
        self._lr = lr

        # Build new model
        self._lstm = TwoOutLSTM_v2(self._input_dim, self._hidden_units, self._layer)

        # Check availability of GPUs
        self._gpu = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        # Adam optimizer
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

        # Cross entropy loss; ignore padding index if provided
        ignore_idx = pad_index if pad_index is not None else -100
        self._loss = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='mean')

    def build(self, name=None):
        """Build new model or load model by name"""

        if (name is None):
            self._lstm = TwoOutLSTM_v2(self._input_dim, self._hidden_units, self._layer)

        else:
            ckpt_path = name + '.dat'

            try:
                # Try weights/state-dict load first
                sd = None
                try:
                    try:
                        weights = torch.load(ckpt_path, map_location=self._device, weights_only=True)
                    except TypeError:
                        weights = torch.load(ckpt_path, map_location=self._device)
                    except Exception:
                        weights = None

                    if weights is not None:
                        if isinstance(weights, dict):
                            sd = weights.get('state_dict', weights.get('model_state_dict', weights))
                        else:
                            sd = weights
                except Exception:
                    sd = None

                if sd is not None:
                    model = TwoOutLSTM_v2(self._input_dim, self._hidden_units, self._layer)
                    try:
                        model.load_state_dict(sd)
                    except Exception:
                        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                        model.load_state_dict(new_sd)
                    self._lstm = model
                else:
                    # Try full-object load, making class importable for unpickling
                    try:
                        import sys, importlib.util, os as _os
                        mod_name = 'two_out_lstm_v2'
                        if mod_name not in sys.modules:
                            mod_path = _os.path.join(_os.path.dirname(__file__), 'two_out_lstm_v2.py')
                            try:
                                spec = importlib.util.spec_from_file_location(mod_name, mod_path)
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                sys.modules[mod_name] = mod
                                _mod = mod
                            except Exception:
                                try:
                                    import two_out_lstm_v2 as _mod
                                except Exception:
                                    _mod = None
                        else:
                            _mod = sys.modules.get(mod_name)

                        safe_list = []
                        if _mod is not None:
                            try:
                                safe_list.append(_mod.TwoOutLSTM_v2)
                            except Exception:
                                pass

                        if safe_list and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
                            with torch.serialization.safe_globals(safe_list):
                                self._lstm = torch.load(ckpt_path, map_location=self._device, weights_only=False)
                        else:
                            try:
                                self._lstm = torch.load(ckpt_path, map_location=self._device)
                            except Exception:
                                raise
                    except Exception as exc_full:
                        print(f"Warning: failed to load checkpoint '{ckpt_path}': {exc_full}")
                        print("Falling back to a freshly initialized TwoOutLSTM_v2 model.")
                        self._lstm = TwoOutLSTM_v2(self._input_dim, self._hidden_units, self._layer)
            except Exception as exc_outer:
                print(f"Warning: unexpected error loading checkpoint '{ckpt_path}': {exc_outer}")
                print("Falling back to a freshly initialized TwoOutLSTM_v2 model.")
                self._lstm = TwoOutLSTM_v2(self._input_dim, self._hidden_units, self._layer)

        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

    def train(self, data, label, epochs=1, batch_size=1):
        '''Train the model
        :param  data:   data array (n_samples, molecule_length, encoding_length)
        :param label:   label array (n_samples, molecule_length)
        :param  epochs: number of epochs for the training
        :param batch_size:  batch size for training
        :return statistic:  array storing computed losses (epochs, batch)
        '''

        # Compute tensor of labels
        label = torch.from_numpy(label).to(self._device)

        # Number of samples
        n_samples = data.shape[0]

        # Calculate number of batches per epoch
        if (n_samples % batch_size) == 0:
            n_iter = n_samples // batch_size
        else:
            n_iter = n_samples // batch_size + 1

        # Store losses
        statistic = np.zeros((epochs, n_iter))

        # Prepare model
        self._lstm.train()

    # Iteration over epochs
        for i in range(epochs):

            # Iteration over batches
            for n in range(n_iter):

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Prepare data with two tokens as input
                data_batch = self.prepare_data(data[batch_start:batch_end])

                # Change axes from (n_samples, molecule_size//2+1, 2*encoding_dim)
                # to (molecule_size//2+1, n_samples, 2*encoding_dim)
                data_batch = np.swapaxes(data_batch, 0, 1)
                data_batch = torch.from_numpy(data_batch).to(self._device)

                # Initialize loss for molecule
                molecule_loss = torch.zeros(1, device=self._device)

                # Track per-step individual losses for debugging (forward, back per step)
                per_step_losses = []

                # Whether any computed loss requires grad (if all steps are pad-only we skip backward)
                grad_needed = False

                # Reset model with correct batch size
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                # Iteration over molecules
                mid = (self._molecule_size - 1) // 2
                for j in range(self._molecule_size // 2):
                    # Prepare input tensor with dimension (1,batch_size, 2*molecule_size)
                    input = data_batch[j].view(1, batch_end - batch_start, -1)

                    # Probabilities for forward and backward token
                    forward, back = self._lstm(input)

                    # Mean cross-entropy loss forward prediction
                    # Targets for this step
                    tgt_f = label[batch_start:batch_end, mid + 1 + j]
                    tgt_b = label[batch_start:batch_end, mid - j]

                    # Compute forward loss unless all targets are pad
                    if (self._pad_index is not None) and torch.all(tgt_f == self._pad_index):
                        loss_forward = torch.zeros(1, device=self._device)
                    else:
                        loss_forward = self._loss(forward.view(batch_end - batch_start, -1), tgt_f)
                        grad_needed = True

                    # Compute backward loss unless all targets are pad
                    if (self._pad_index is not None) and torch.all(tgt_b == self._pad_index):
                        loss_back = torch.zeros(1, device=self._device)
                    else:
                        loss_back = self._loss(back.view(batch_end - batch_start, -1), tgt_b)
                        grad_needed = True

                    # Add losses from both sides
                    loss_tot = torch.add(loss_forward, loss_back)

                    # Add to molecule loss (use add so grad flows if any loss has grad)
                    molecule_loss = torch.add(molecule_loss, loss_tot)

                    # Record individual losses (forward, back)
                    try:
                        per_step_losses.append(float(loss_forward.item()))
                    except Exception:
                        per_step_losses.append(None)
                    try:
                        per_step_losses.append(float(loss_back.item()))
                    except Exception:
                        per_step_losses.append(None)

                    # Quick numeric checks and diagnostics
                    try:
                        if torch.isnan(loss_forward) or torch.isinf(loss_forward) or torch.isnan(loss_back) or torch.isinf(loss_back):
                            print(f"Numeric issue in FBRNN: epoch={i} batch={n} step={j} loss_f={loss_forward} loss_b={loss_back}", flush=True)
                            print(f" forward pred min={float(forward.min().item())} max={float(forward.max().item())} mean={float(forward.mean().item())}", flush=True)
                            print(f" back pred min={float(back.min().item())} max={float(back.max().item())} mean={float(back.mean().item())}", flush=True)
                            try:
                                print(f" tgt_f unique={torch.unique(tgt_f).cpu().numpy()} tgt_b unique={torch.unique(tgt_b).cpu().numpy()}", flush=True)
                            except Exception:
                                pass
                            raise RuntimeError("Encountered NaN/Inf loss in FBRNN training")
                    except Exception:
                        pass

                # Compute backpropagation if needed (skip if all steps were pad-only)
                if grad_needed:
                    self._optimizer.zero_grad()
                    molecule_loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(self._lstm.parameters(), max_norm=5.0)
                    except Exception:
                        pass
                    # Store statistics: loss per token (middle token not included)
                    statistic[i, n] = molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)

                    # Print per-step individual losses for debugging (forward/back alternation)
                    # print(f"Train(epoch={i}) batch={n} mean_loss={statistic[i,n]:.6f} per_step_losses={per_step_losses}", flush=True)

                    # Perform optimization step and reset gradients
                    self._optimizer.step()
                else:
                    # No gradients to backpropagate (all steps were pad-only)
                    statistic[i, n] = 0.0
                    # print(f"Train(epoch={i}) batch={n} mean_loss={statistic[i,n]:.6f} per_step_losses={per_step_losses} (skipped backward)", flush=True)

        return statistic

    def validate(self, data, label, batch_size=128):
        ''' Validation of model and compute error
        :param data:    test data (n_samples, molecule_size, encoding_size)
        :param label:   label data (n_samples, molecule_size)
        :param batch_size:  batch size for validation
        :return:        mean loss over test data
        '''

        # Use train mode to get loss consistent with training
        self._lstm.train()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Compute tensor of labels
            label = torch.from_numpy(label).to(self._device)

            # Number of samples
            n_samples = data.shape[0]

            # Initialize loss for molecule
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

                # Prepare data with two tokens as input
                data_batch = self.prepare_data(data[batch_start:batch_end])

                # Change axes from (n_samples, molecule_size//2+1, 2*encoding_dim)
                # to (molecule_size//2+1, n_samples, 2*encoding_dim)
                data_batch = np.swapaxes(data_batch, 0, 1)
                data_batch = torch.from_numpy(data_batch).to(self._device)

                # Initialize loss for molecule at correct device
                molecule_loss = torch.zeros(1).to(self._device)

                # Reset model with correct batch size and device
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                mid = (self._molecule_size - 1) // 2
                for j in range(self._molecule_size // 2):
                    # Prepare input tensor with dimension (1,n_samples, 2*molecule_size)
                    input = data_batch[j].view(1, batch_end - batch_start, -1)

                    # Forward and backward output
                    forward, back = self._lstm(input)

                    # Mean cross-entropy loss forward prediction
                    loss_forward = self._loss(forward.view(batch_end - batch_start, -1),
                                              label[batch_start:batch_end, mid + 1 + j])

                    # Mean Cross-entropy loss backward prediction
                    loss_back = self._loss(back.view(batch_end - batch_start, -1),
                                           label[batch_start:batch_end, mid - j])

                    # Add losses from both sides
                    loss_tot = torch.add(loss_forward, loss_back)

                    # Add to molecule loss
                    molecule_loss = torch.add(molecule_loss, loss_tot)

                tot_loss += molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)
        return tot_loss / n_iter

    def sample(self, seed_token, T=1):
        '''Generate new molecule
        :param seed_token: starting token or sequence. Accepts shape (vocab,) or (L, vocab).
        :param T:          sampling temperature
        :return molecule:  (1, molecule_size, vocab)
        '''

        # Prepare model
        self._lstm.eval()

        # Normalize seed to shape (L, vocab)
        seed = np.asarray(seed_token)
        vocab = self._input_dim // 2
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != vocab:
            raise ValueError(f"FBRNN.sample: seed has wrong vocab dim {seed.shape[1]} != {vocab}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        # Build molecule canvas (kept in numpy for convenience)
        molecule = np.zeros((1, self._molecule_size, vocab), dtype=np.float32)

        # Determine centered placement
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
        molecule[0, start:end, :] = seed

        # Prime hidden state and generate with gradients disabled
        with torch.no_grad():
            # Step 0: duplicate middle token
            center = molecule[0, mid, :]
            in0 = np.concatenate([center, center], axis=0)
            input_vec = torch.from_numpy(in0.astype(np.float32)).view(1, 1, -1).to(self._device)
            self._lstm.new_sequence(device=self._device)
            _ = self._lstm(input_vec)

            # Subsequent known steps j use pair (mid+1+j, mid-j) as long as both lie within the seeded window
            j = 0
            while True:
                pos_r = mid + 1 + j
                pos_l = mid - j
                if pos_r < end and pos_l >= start:
                    pair = np.concatenate([molecule[0, pos_r, :], molecule[0, pos_l, :]], axis=0)
                    input_vec = torch.from_numpy(pair.astype(np.float32)).view(1, 1, -1).to(self._device)
                    _ = self._lstm(input_vec)
                    j += 1
                else:
                    break

            # Initialize contexts at current frontiers for generation
            f_ctx = molecule[0, end - 1, :].copy()
            b_ctx = molecule[0, start, :].copy()
            input_vec = torch.from_numpy(np.concatenate([f_ctx, b_ctx], axis=0).astype(np.float32)).view(1, 1, -1).to(self._device)

            # Remaining tokens to fill
            total_to_fill = self._molecule_size - L
            filled = 0
            while filled < total_to_fill:
                forward, back = self._lstm(input_vec)

                new_f_ctx = f_ctx
                new_b_ctx = b_ctx

                # Predict right side if there is room
                if end < self._molecule_size:
                    token_f = self.sample_token(np.squeeze(forward.cpu().detach().numpy()), T)
                    molecule[0, end, token_f] = 1.0
                    new_f_ctx = molecule[0, end, :].copy()
                    end += 1
                    filled += 1

                # Predict at current left frontier if there is room
                if start > 0 and filled < total_to_fill:
                    token_b = self.sample_token(np.squeeze(back.cpu().detach().numpy()), T)
                    molecule[0, start - 1, token_b] = 1.0
                    new_b_ctx = molecule[0, start - 1, :].copy()
                    start -= 1
                    filled += 1

                # Prepare next input
                f_ctx, b_ctx = new_f_ctx, new_b_ctx
                input_vec = torch.from_numpy(np.concatenate([f_ctx, b_ctx], axis=0).astype(np.float32)).view(1, 1, -1).to(self._device)

        return molecule

    def sample_with_logp(self, seed_token, T=1.0):
        """Generate a molecule and return per-position log-probabilities of the taken actions.

        Returns:
          molecule: np.ndarray of shape (1, molecule_size, vocab)
          log_probs: torch.Tensor of shape (molecule_size,) with log p(action) at positions where an action was taken, 0 elsewhere
          mask: torch.Tensor of shape (molecule_size,) with 1.0 at positions where an action was taken, 0 elsewhere
        """
        # Enable training mode so RNNs allow backprop through time
        self._lstm.train()

        # Normalize seed to shape (L, vocab)
        seed = np.asarray(seed_token)
        vocab = self._input_dim // 2
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != vocab:
            raise ValueError(f"FBRNN.sample_with_logp: seed has wrong vocab dim {seed.shape[1]} != {vocab}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        with torch.no_grad():
            # Molecule (1, size, vocab)
            molecule = np.zeros((1, self._molecule_size, vocab), dtype=np.float32)

            # Center placement
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
            molecule[0, start:end, :] = seed

        # Initialize log-prob buffers (torch tensors to carry gradients)
        log_probs = torch.zeros((self._molecule_size,), dtype=torch.float32, device=self._device)
        mask = torch.zeros((self._molecule_size,), dtype=torch.float32, device=self._device)

        # Prime hidden state with center token pair and any known pairs
        center = molecule[0, mid, :]
        in0 = np.concatenate([center, center], axis=0)
        input_vec = torch.from_numpy(in0.astype(np.float32)).view(1, 1, -1).to(self._device)
        self._lstm.new_sequence(device=self._device)
        _ = self._lstm(input_vec)

        j = 0
        while True:
            pos_r = mid + 1 + j
            pos_l = mid - j
            if pos_r < end and pos_l >= start:
                pair = np.concatenate([molecule[0, pos_r, :], molecule[0, pos_l, :]], axis=0)
                input_vec = torch.from_numpy(pair.astype(np.float32)).view(1, 1, -1).to(self._device)
                _ = self._lstm(input_vec)
                j += 1
            else:
                break

        # Initialize contexts at frontiers
        f_ctx = molecule[0, end - 1, :].copy()
        b_ctx = molecule[0, start, :].copy()
        input_vec = torch.from_numpy(np.concatenate([f_ctx, b_ctx], axis=0).astype(np.float32)).view(1, 1, -1).to(self._device)

        # Fill remaining positions while capturing log-probs of taken actions
        total_to_fill = self._molecule_size - L
        filled = 0
        while filled < total_to_fill:
            forward, back = self._lstm(input_vec)

            new_f_ctx = f_ctx
            new_b_ctx = b_ctx

            # Right action
            if end < self._molecule_size:
                logits_f = (forward[0, 0, :] / float(T))
                logp_f = torch.log_softmax(logits_f, dim=-1)
                probs_f = torch.softmax(logits_f, dim=-1)
                tok_f = torch.multinomial(probs_f, num_samples=1).item()
                molecule[0, end, tok_f] = 1.0
                new_f_ctx = molecule[0, end, :].copy()
                log_probs[end] = logp_f[tok_f]
                mask[end] = 1.0
                end += 1
                filled += 1

            # Left action
            if start > 0 and filled < total_to_fill:
                logits_b = (back[0, 0, :] / float(T))
                logp_b = torch.log_softmax(logits_b, dim=-1)
                probs_b = torch.softmax(logits_b, dim=-1)
                tok_b = torch.multinomial(probs_b, num_samples=1).item()
                molecule[0, start - 1, tok_b] = 1.0
                new_b_ctx = molecule[0, start - 1, :].copy()
                log_probs[start - 1] = logp_b[tok_b]
                mask[start - 1] = 1.0
                start -= 1
                filled += 1

            f_ctx, b_ctx = new_f_ctx, new_b_ctx
            input_vec = torch.from_numpy(np.concatenate([f_ctx, b_ctx], axis=0).astype(np.float32)).view(1, 1, -1).to(self._device)

        return molecule, log_probs, mask


    def prepare_data(self, data):
        '''Reshape data to get two tokens as single input
        :params data:           data array (n_samples, molecule_length, encoding_length)
        :return cominde_input:  reshape data (n_samples, molecule_size//2 +1, 2*encoding_length)
        '''

        # Number of samples
        n_samples = data.shape[0]

        # Reshaped data array
        combined_input = np.zeros((n_samples, self._molecule_size // 2 + 1, self._input_dim)).astype(np.float32)

        mid = (self._molecule_size - 1) // 2
        for i in range(n_samples):
            # First Input is two times the token in the middle
            combined_input[i, 0, :self._input_dim // 2] = data[i, mid, :]
            combined_input[i, 0, self._input_dim // 2:] = data[i, mid, :]

            # Merge two tokens to a single input
            for j in range(self._molecule_size // 2):
                combined_input[i, j + 1, :self._input_dim // 2] = data[i, mid + 1 + j, :]
                combined_input[i, j + 1, self._input_dim // 2:] = data[i, mid - j, :]

        return combined_input

    def sample_token(self, out, T=1.0):
        ''' Sample token
        :param out: output values from model
        :param T:   sampling temperature
        :return:    index of predicted token
        '''

        # Explicit conversion to float64 avoiding truncation errors
        out = out.astype('float64')

        # Compute probabilities with specific temperature
        out_T = out / T - max(out / T)
        p = np.exp(out_T) / np.sum(np.exp(out_T))

        # Generate new token at random
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def save(self, name='test_model'):
        torch.save(self._lstm, name + '.dat')
