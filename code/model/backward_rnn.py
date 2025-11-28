"""
BackwardRNN: same semantics as ForwardRNN but runs from the end and predicts the previous token.
"""

import numpy as np
import torch
import torch.nn as nn
from one_out_lstm import OneOutLSTM

# torch.manual_seed(1)
# np.random.seed(5)


class BackwardRNN():

    def __init__(self, molecule_size=7, encoding_dim=55, lr=.01, hidden_units=256, pad_index=None):

        self._molecule_size = molecule_size
        self._input_dim = encoding_dim
        self._layer = 2
        self._hidden_units = hidden_units
        self._pad_index = pad_index

        # Learning rate
        self._lr = lr

        # Build new model
        self._lstm = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)

        # Check availability of GPUs
        self._gpu = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        # Create optimizer after the model is initialized (and moved to device)
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))
        # Cross entropy loss; ignore padding index if provided
        ignore_idx = self._pad_index if self._pad_index is not None else -100
        self._loss = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='mean')


    def build(self, name=None):
        """Build new model or load model by name"""
        if (name is None):
            self._lstm = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)

        else:
            ckpt_path = name + '.dat'
            try:
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
                    model = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)
                    try:
                        model.load_state_dict(sd)
                    except Exception:
                        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                        model.load_state_dict(new_sd)
                    self._lstm = model
                else:
                    # Try full-object load with safe globals
                    try:
                        import sys, importlib.util, os as _os
                        mod_name = 'one_out_lstm'
                        if mod_name not in sys.modules:
                            mod_path = _os.path.join(_os.path.dirname(__file__), 'one_out_lstm.py')
                            try:
                                spec = importlib.util.spec_from_file_location(mod_name, mod_path)
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                sys.modules[mod_name] = mod
                                _mod = mod
                            except Exception:
                                try:
                                    import one_out_lstm as _mod
                                except Exception:
                                    _mod = None
                        else:
                            _mod = sys.modules.get(mod_name)

                        safe_list = []
                        if _mod is not None:
                            try:
                                safe_list.append(_mod.OneOutLSTM)
                            except Exception:
                                pass

                        if safe_list and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
                            with torch.serialization.safe_globals(safe_list):
                                self._lstm = torch.load(ckpt_path, map_location=self._device, weights_only=False)
                        else:
                            self._lstm = torch.load(ckpt_path, map_location=self._device)
                    except Exception as exc_full:
                        print(f"Warning: failed to load checkpoint '{ckpt_path}': {exc_full}")
                        print("Falling back to a freshly initialized OneOutLSTM model.")
                        self._lstm = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)
            except Exception as exc_outer:
                print(f"Warning: unexpected error loading checkpoint '{ckpt_path}': {exc_outer}")
                print("Falling back to a freshly initialized OneOutLSTM model.")
                self._lstm = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)

        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        # Recreate optimizer for the (possibly new) model parameters
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))


    def train(self, data, label, epochs, batch_size):
        """Train on data reversed so BackwardRNN predicts previous token from the end.
        data: (n_samples, molecule_size, encoding_dim)
        label: (n_samples, molecule_size)
        """

        # Reverse molecules along sequence axis so we can train like ForwardRNN
        data_rev = np.flip(data, axis=1).copy()
        label_rev = np.flip(label, axis=1).copy()

        # Delegate to forward-like training on reversed sequences (implementation similar to ForwardRNN)
        # Number of samples
        n_samples = data_rev.shape[0]

        # Change axes from (n_samples, molecule_size, encoding_dim) to (molecule_size, n_samples, encoding_dim)
        data_t = np.swapaxes(data_rev, 0, 1)

        # Compute tensor of labels
        label_t = torch.from_numpy(label_rev).to(self._device)

        # Calculate number of batches per epoch
        if (n_samples % batch_size) == 0:
            n_iter = n_samples // batch_size
        else:
            n_iter = n_samples // batch_size + 1

        statistic = np.zeros((epochs, n_iter))

        self._lstm.train()

        for i in range(epochs):
            for n in range(n_iter):
                self._optimizer.zero_grad()
                losses = []

                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                self._lstm.new_sequence(batch_end - batch_start, self._device)

                batch_data = torch.from_numpy(data_t[:, batch_start:batch_end, :].astype('float32')).to(self._device)

                molecule_loss = torch.zeros(1).to(self._device)
                for j in range(self._molecule_size - 1):
                    # Feed exactly one reversed timestep and predict next (in reversed space)
                    input_t = batch_data[j:j+1, :, :]  # shape (1, batch, dim)
                    pred_seq = self._lstm.forward(input_t)
                    pred_view = pred_seq[-1, :, :]

                    step_labels = label_t[batch_start:batch_end, j + 1]
                    if (self._pad_index is not None) and torch.all(step_labels == self._pad_index):
                        loss = torch.zeros(1).to(self._device)
                    else:
                        loss = self._loss(pred_view, step_labels)

                    losses.append(loss.item())
                    molecule_loss = torch.add(molecule_loss, loss)

                if len(losses) == 0:
                    statistic[i, n] = 0.0
                    continue

                molecule_loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(self._lstm.parameters(), max_norm=5.0)
                except Exception:
                    pass

                statistic[i, n] = np.sum(losses) / (self._molecule_size - 1)
                # print(f"BackwardTrain epoch={i} batch={n} mean_loss={statistic[i,n]:.6f} per_step_losses={losses}", flush=True)

                self._optimizer.step()

        return statistic


    def validate(self, data, label):
        # Reverse sequences then run forward-like validate
        data_rev = np.flip(data, axis=1).copy()
        label_rev = np.flip(label, axis=1).copy()

        # Use train mode to get consistent loss
        self._lstm.train()
        with torch.no_grad():
            label_t = torch.from_numpy(label_rev).to(self._device)
            n_samples = data_rev.shape[0]
            data_t = np.swapaxes(data_rev, 0, 1).astype('float32')
            data_t = torch.from_numpy(data_t).to(self._device)

            molecule_loss = torch.zeros(1).to(self._device)
            self._lstm.new_sequence(n_samples, self._device)

            for j in range(self._molecule_size - 1):
                input = data_t[j, :, :].view(1, n_samples, -1)
                pred_seq = self._lstm.forward(input)
                forward = pred_seq[-1, :, :]
                step_labels = label_t[:, j + 1]
                if (self._pad_index is not None) and torch.all(step_labels == self._pad_index):
                    loss_forward = torch.zeros(1).to(self._device)
                else:
                    loss_forward = self._loss(forward, step_labels)
                molecule_loss = torch.add(molecule_loss, loss_forward)

        return molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)


    def sample(self, seed_token, T=1):
        """Sample by operating in reversed sequence space.
        Steps:
        - Reverse the seed order
        - Prime the hidden state by feeding all seed tokens (except last) in reversed space
        - Generate tokens to the right in reversed space
        - Flip back to original orientation before returning
        """
        self._lstm.eval()
        seed = np.asarray(seed_token)
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != self._input_dim:
            raise ValueError(f"BackwardRNN.sample: seed has wrong vocab dim {seed.shape[1]} != {self._input_dim}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        with torch.no_grad():
            # Operate in reversed space: rev_output[t] corresponds to original position size-1-t
            rev_output = np.zeros((self._molecule_size, self._input_dim), dtype=np.float32)
            # Reverse the seed order for reversed space
            rev_seed = seed[::-1].copy()
            rev_output[:L, :] = rev_seed

            # Prime hidden state by feeding rev_seed[:-1]
            self._lstm.new_sequence(batch_size=1, device=self._device)
            for k in range(max(0, L - 1)):
                inp_k = torch.from_numpy(rev_output[k, :].astype(np.float32)).view(1, 1, -1).to(self._device)
                _ = self._lstm.forward(inp_k)

            # Start from the last seed token in reversed space
            input = torch.from_numpy(rev_output[L - 1, :].astype(np.float32)).view(1, 1, -1).to(self._device)

            # Generate tokens to the right in reversed space
            for pos in range(L, self._molecule_size):
                pred_seq = self._lstm.forward(input)
                forward = pred_seq[-1, :, :]
                token = self.sample_token(forward.cpu().detach().numpy().reshape(-1), T)
                rev_output[pos, token] = 1.0
                input = torch.from_numpy(rev_output[pos, :].astype(np.float32)).view(1, 1, -1).to(self._device)

            # Flip back to original orientation before returning
            molecule = np.zeros((1, self._molecule_size, self._input_dim), dtype=np.float32)
            molecule[0, :, :] = rev_output[::-1, :]
        return molecule

    def sample_with_logp(self, seed_token, T=1.0):
        """Generate a molecule and return per-position log-probabilities of the taken actions.

        For BackwardRNN, generation is performed in reversed space; we map the log-probs
        back to the original orientation before returning.

        Returns:
          molecule: np.ndarray of shape (1, molecule_size, vocab)
          log_probs: torch.Tensor of shape (molecule_size,) with log p(action) at positions where an action was taken, 0 elsewhere
          mask: torch.Tensor of shape (molecule_size,) with 1.0 at positions where an action was taken, 0 elsewhere
        """
        self._lstm.train()
        seed = np.asarray(seed_token)
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != self._input_dim:
            raise ValueError(f"BackwardRNN.sample_with_logp: seed has wrong vocab dim {seed.shape[1]} != {self._input_dim}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        device = self._device
        # reversed space buffers as tensors
        rev_output_t = torch.zeros((self._molecule_size, self._input_dim), dtype=torch.float32, device=device)
        rev_seed = torch.from_numpy(seed[::-1].copy().astype(np.float32)).to(device)
        rev_output_t[:L, :] = rev_seed

        # Prime hidden state by feeding rev_seed[:-1]
        self._lstm.new_sequence(batch_size=1, device=device)
        for k in range(max(0, L - 1)):
            inp_k = rev_output_t[k, :].clone().view(1, 1, -1)
            _ = self._lstm.forward(inp_k)

        input_t = rev_output_t[L - 1, :].clone().view(1, 1, -1)

        taken_positions_orig = []
        taken_logps = []

        for pos_rev in range(L, self._molecule_size):
            logits_seq = self._lstm.forward(input_t)  # (1, 1, vocab)
            logits1d = logits_seq[-1, 0, :].view(-1)
            probs = torch.softmax(logits1d / float(T), dim=0)
            token_idx = torch.multinomial(probs, num_samples=1)[0].item()
            logp_taken = torch.log_softmax(logits1d / float(T), dim=0)[token_idx]

            # Write into reversed buffer
            rev_output_t[pos_rev, :] = 0.0
            rev_output_t[pos_rev, token_idx] = 1.0
            input_t = rev_output_t[pos_rev, :].clone().view(1, 1, -1)

            # Map to original orientation index
            pos_orig = self._molecule_size - 1 - pos_rev
            taken_positions_orig.append(int(pos_orig))
            taken_logps.append(logp_taken)

        if len(taken_positions_orig) > 0:
            idx_t = torch.tensor(taken_positions_orig, dtype=torch.long, device=device)
            src = torch.stack(taken_logps, dim=0)
            log_probs = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device).scatter(0, idx_t, src)
            mask = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device).scatter(0, idx_t, torch.ones_like(src))
        else:
            log_probs = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device)
            mask = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device)

        molecule = np.zeros((1, self._molecule_size, self._input_dim), dtype=np.float32)
        molecule[0, :, :] = rev_output_t.detach().cpu().numpy()[::-1, :]
        return molecule, log_probs, mask

    def sample_token(self, out, T=1.0):
        out = out.astype('float64')
        p = np.exp(out / T) / np.sum(np.exp(out / T))
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def save(self, name='backward_model'):
        torch.save(self._lstm, name + '.dat')
