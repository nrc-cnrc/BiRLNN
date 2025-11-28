"""
Implementation of one-directional RNN for SMILES generation
"""

import numpy as np
import torch
import torch.nn as nn
from one_out_lstm import OneOutLSTM
import torch.nn.functional as F
from scipy.special import logsumexp

# torch.manual_seed(1)
# np.random.seed(5)


class ForwardRNN():

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

        # Recreate optimizer for the (possibly new) model parameters so
        # optimizer.step() updates the current `self._lstm` parameters.
        # Without this line, if build() is called multiple times,
        # the optimizer would keep updating the parameters of the
        # original model, not the newly created one.
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))


    def train(self, data, label, epochs, batch_size):
        '''Train the model
        :param  data:   data array (n_samples, molecule_size, encoding_length)
        :param  label:  label array (n_samples, molecule_size)
        :param  epochs: number of epochs for training
        :param  batch_size: batch_size for training
        :return statistic:  array storing computed losses (epochs, batch)
        '''

        # Number of samples
        n_samples = data.shape[0]

        # Change axes from (n_samples, molecule_size, encoding_dim) to (molecule_size, n_samples, encoding_dim)
        data = np.swapaxes(data, 0, 1)
        
        # Compute tensor of labels
        label = torch.from_numpy(label).to(self._device)

        # We'll slice this per-batch below and move to device

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

                # Set gradient to zero for batch (match BIMODAL style)
                self._optimizer.zero_grad()

                # Store losses in list
                losses = []

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Reset model hidden state once per sequence (batch)
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                # Current batch
                batch_data = torch.from_numpy(data[:, batch_start:batch_end, :].astype('float32')).to(self._device)

                # Use a torch scalar to accumulate losses for backward once
                molecule_loss = torch.zeros(1).to(self._device)
                for j in range(self._molecule_size - 1):
                    # Feed exactly one timestep (j) and predict next token (j+1)
                    input_t = batch_data[j:j+1, :, :]  # shape (1, batch, dim)
                    pred_seq = self._lstm.forward(input_t)  # shape (1, batch, vocab)
                    pred_view = pred_seq[-1, :, :]        # shape (batch, vocab)

                    # Compute loss against labels at position j+1
                    step_labels = label[batch_start:batch_end, j + 1]
                    if (self._pad_index is not None) and torch.all(step_labels == self._pad_index):
                        # all labels are padding for this step -> skip (loss 0)
                        loss = torch.zeros(1).to(self._device)
                    else:
                        loss = self._loss(pred_view, step_labels)

                    # Append loss of current position
                    losses.append(loss.item())

                    # Quick pred stats for debugging
                    try:
                        pred_min = float(pred_view.min().item())
                        pred_max = float(pred_view.max().item())
                        pred_mean = float(pred_view.mean().item())
                    except Exception:
                        pred_min = pred_max = pred_mean = None

                    # If loss is NaN/Inf, dump context and abort
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Numeric issue detected: NaN/Inf loss at epoch={i} batch={n} step={j}", flush=True)
                        print(f" pred_min={pred_min} pred_max={pred_max} pred_mean={pred_mean}", flush=True)
                        print(f" step_labels unique={(torch.unique(step_labels).cpu().numpy() if step_labels is not None else None)}", flush=True)
                        raise RuntimeError("Encountered NaN/Inf loss during ForwardRNN training")

                    # Accumulate into molecule_loss tensor
                    molecule_loss = torch.add(molecule_loss, loss)

                # If no prediction steps were performed (molecule_size == 1)
                # then molecule_loss is a plain zero tensor without grad graph.
                # Skip backward/optimization in that case.
                if len(losses) == 0:
                    # Store statistics as zero loss for this batch
                    statistic[i, n] = 0.0
                    # Nothing to backpropagate or optimize
                    continue

                # Backpropagate the summed loss once
                # (optimizer.zero_grad() was already called at batch start)
                molecule_loss.backward()

                # Inspect gradients after backward to detect NaNs/Inf
                try:
                    total_grad_norm_sq = 0.0
                    for p in self._lstm.parameters():
                        if p.grad is not None:
                            gnorm = p.grad.data.norm(2).item()
                            total_grad_norm_sq += gnorm * gnorm
                    total_grad_norm = total_grad_norm_sq ** 0.5
                    if np.isnan(total_grad_norm) or np.isinf(total_grad_norm):
                        print(f"Numeric issue: NaN/Inf gradient norm at epoch={i} batch={n} grad_norm={total_grad_norm}", flush=True)
                except Exception:
                    pass

                # Clip gradients to avoid explosion / NaNs
                try:
                    torch.nn.utils.clip_grad_norm_(self._lstm.parameters(), max_norm=5.0)
                except Exception:
                    pass
                # Print loss information for debugging (epoch, batch, molecule loss, per-step losses)
                # print(f"Train epoch={i} batch={n} molecule_loss={molecule_loss.item():.6f} mean_loss={(np.sum(losses)/(self._molecule_size - 1)):.6f} individual losses={losses}", flush=True)

                # Store statistics: loss per token
                statistic[i, n] = np.sum(losses) / (self._molecule_size - 1)

                # Perform optimization step
                self._optimizer.step()

        return statistic

    def validate(self, data, label):
        ''' Validation of model and compute error
        :param data:    test data (n_samples, molecule_size, encoding_size)
        :return:        mean loss over test data
        '''

        # Use train mode to get loss consistent with training
        self._lstm.train()

        # Gradient is not compute to reduce memory usage
        with torch.no_grad():
            # Compute tensor of labels
            label = torch.from_numpy(label).to(self._device)

            # Number of samples
            n_samples = data.shape[0]

            # Change axes from (n_samples, molecule_size , encoding_dim) to (molecule_size , n_samples, encoding_dim)
            data = np.swapaxes(data, 0, 1).astype('float32')

            # Create tensor for data and store at correct device
            data = torch.from_numpy(data).to(self._device)

            # Initialize loss for molecule at correct device
            molecule_loss = torch.zeros(1).to(self._device)

            # Reset model with correct batch size and device
            self._lstm.new_sequence(n_samples, self._device)

            for j in range(self._molecule_size - 1):
                # Prepare input tensor with dimension (1,n_samples, input_dim)
                input = data[j, :, :].view(1, n_samples, -1)

                # Probabilities for next prediction using one-step forward
                pred_seq = self._lstm.forward(input)
                forward = pred_seq[-1, :, :]

                # Mean cross-entropy loss
                step_labels = label[:, j + 1]
                if (self._pad_index is not None) and torch.all(step_labels == self._pad_index):
                    loss_forward = torch.zeros(1).to(self._device)
                else:
                    loss_forward = self._loss(forward, step_labels)

                # Add to molecule loss
                molecule_loss = torch.add(molecule_loss, loss_forward)

        return molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)

    def sample(self, seed_token, T=1):
        '''Generate new molecule
        :param seed_token: starting token or sequence (vocab,) or (L, vocab)
        :param T:          sampling temperature
        :return molecule:  (1, molecule_length, vocab)
        '''
        # Prepare model
        self._lstm.eval()

        # Normalize seed to shape (L, vocab)
        seed = np.asarray(seed_token)
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != self._input_dim:
            raise ValueError(f"ForwardRNN.sample: seed has wrong vocab dim {seed.shape[1]} != {self._input_dim}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        # Gradient is not compute to reduce memory usage
        with torch.no_grad():
            output = np.zeros((self._molecule_size, self._input_dim), dtype=np.float32)
            molecule = np.zeros((1, self._molecule_size, self._input_dim), dtype=np.float32)

            # Pre-fill seed at the beginning (left-aligned for forward generation)
            output[:L, :] = seed
            molecule[0, :L, :] = seed

            # Prime hidden state by feeding the seed tokens sequentially (except last)
            self._lstm.new_sequence(batch_size=1, device=self._device)
            for k in range(max(0, L - 1)):
                inp_k = torch.from_numpy(output[k, :].astype(np.float32)).view(1, 1, -1).to(self._device)
                _ = self._lstm.forward(inp_k)  # advance state; ignore prediction

            # Start from the last seed token
            input = torch.from_numpy(output[L - 1, :].astype(np.float32)).view(1, 1, -1).to(self._device)

            # Sample remaining tokens
            for j in range(L - 1, self._molecule_size - 1):
                pred_seq = self._lstm.forward(input)
                forward = pred_seq[-1, :, :]
                token_forward = self.sample_token(forward.cpu().detach().numpy().reshape(-1), T)
                molecule[0, j + 1, token_forward] = 1.0
                output[j + 1, token_forward] = 1.0
                input = torch.from_numpy(output[j + 1, :].astype(np.float32)).view(1, 1, -1).to(self._device)

        return molecule

    def sample_with_logp(self, seed_token, T=1.0):
        """Generate a molecule and return per-position log-probabilities of the taken actions.

        Returns:
          molecule: np.ndarray of shape (1, molecule_size, vocab)
          log_probs: torch.Tensor of shape (molecule_size,) with log p(action) at positions where an action was taken, 0 elsewhere
          mask: torch.Tensor of shape (molecule_size,) with 1.0 at positions where an action was taken, 0 elsewhere
        """
        # Enable training mode so the LSTM creates a computation graph for logits
        self._lstm.train()

        # Normalize seed to shape (L, vocab)
        seed = np.asarray(seed_token)
        if seed.ndim == 1:
            seed = seed.reshape(1, -1)
        if seed.shape[1] != self._input_dim:
            raise ValueError(f"ForwardRNN.sample_with_logp: seed has wrong vocab dim {seed.shape[1]} != {self._input_dim}")
        L = seed.shape[0]
        if L > self._molecule_size:
            raise ValueError(f"Seed length {L} exceeds molecule_size {self._molecule_size}")

        device = self._device
        # Tensor buffers on device
        output_t = torch.zeros((self._molecule_size, self._input_dim), dtype=torch.float32, device=device)
        output_t[:L, :] = torch.from_numpy(seed.astype(np.float32)).to(device)

        # Prime hidden state by feeding the seed tokens sequentially (except last)
        self._lstm.new_sequence(batch_size=1, device=device)
        for k in range(max(0, L - 1)):
            inp_k = output_t[k, :].clone().view(1, 1, -1)
            _ = self._lstm.forward(inp_k)

        # Start from the last seed token
        input_t = output_t[L - 1, :].clone().view(1, 1, -1)

        taken_positions = []
        taken_logps = []

        # Sample remaining tokens and record log-probabilities
        for pos in range(L, self._molecule_size):
            logits_seq = self._lstm.forward(input_t)  # shape (1, 1, vocab)
            logits1d = logits_seq[-1, 0, :].view(-1)

            # Sample token using temperature
            probs = torch.softmax(logits1d / float(T), dim=0)
            token_idx = torch.multinomial(probs, num_samples=1)[0].item()

            # Log-prob of taken action
            logp_taken = torch.log_softmax(logits1d / float(T), dim=0)[token_idx]
            taken_positions.append(int(pos))
            taken_logps.append(logp_taken)

            # Write one-hot token and update input
            output_t[pos, :] = 0.0
            output_t[pos, token_idx] = 1.0
            input_t = output_t[pos, :].clone().view(1, 1, -1)

        if len(taken_positions) > 0:
            idx_t = torch.tensor(taken_positions, dtype=torch.long, device=device)
            src = torch.stack(taken_logps, dim=0)
            log_probs = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device).scatter(0, idx_t, src)
            mask = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device).scatter(0, idx_t, torch.ones_like(src))
        else:
            log_probs = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device)
            mask = torch.zeros((self._molecule_size,), dtype=torch.float32, device=device)

        molecule = output_t.detach().cpu().numpy().reshape(1, self._molecule_size, self._input_dim)
        return molecule, log_probs, mask
    
    def sample_token(self, out, T=1.0):
        ''' Sample token
        :param out: output values from model
        :param T:   sampling temperature
        :return:    index of predicted token
        '''
        # Explicit conversion to float64 avoiding truncation errors
        out = out.astype('float64')

        # Compute probabilities with specific temperature
        p = np.exp(out / T) / np.sum(np.exp(out / T))

        # Generate new token at random
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def save(self, name='test_model'):
        torch.save(self._lstm, name + '.dat')
