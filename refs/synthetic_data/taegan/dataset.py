import json
import os
from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import DataLoader, TensorDataset


class TrainingDataLoader(DataLoader):
    def __init__(self,
                 cache_dir: str = "diabetes-demo",
                 batch_size: int = 500):
        self.data = torch.load(os.path.join(cache_dir, "train-data.pt"))
        with open(os.path.join(cache_dir, "span-info.json"), "r") as f:
            self.span_info = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = torch.zeros(self.data.shape[0], len(self.span_info), dtype=torch.float, device=device)
        discretizers = {}
        st = 0
        for i, (w, t) in enumerate(self.span_info):
            col = self.data[:, st:st+w]
            if t == "onehot":
                cnt = col.sum(dim=0).long()
                weights = self._calculate_weights(cnt)
                for j in range(w):
                    self.weights[col[:, j].bool(), i] = weights[j]
            elif t == "normal":
                discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
                discretizers[i] = discretizer
                discretized = discretizer.fit_transform(col.cpu().numpy())
                discretized = torch.from_numpy(discretized).to(device, dtype=torch.long)
                oh = torch.nn.functional.one_hot(discretized[:, 0], discretizer.n_bins_[0])
                cnt = oh.sum(dim=0).long()
                weights = self._calculate_weights(cnt)
                for j in range(oh.shape[-1]):
                    self.weights[oh[:, j].bool(), i] = weights[j]
            else:
                raise ValueError(f"Type {t} is invalid.")
            st += w
        self.weights = self.weights / self.weights.sum(dim=0).unsqueeze(0)
        self.weights = self.weights.cpu()

        size = int(np.ceil(self.data.shape[0] / batch_size) * batch_size)
        super().__init__(
            TensorDataset(self.data[[0] * size, :0]), batch_size=batch_size, collate_fn=self._collate
        )

    @staticmethod
    def _calculate_weights(counts: torch.LongTensor) -> torch.FloatTensor:
        weights = torch.log(counts + 1)  # smooth + 1
        weights = weights / weights.sum(dim=0).unsqueeze(0)
        per_row_weights = weights / counts
        per_row_weights[counts == 0] = 0
        return per_row_weights

    def _collate(self, batch: List) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        batch_size = len(batch)
        data = []
        mask = []
        for _ in range(batch_size):
            d, m = self._next_data()
            data.append(d)
            mask.append(m)
        return torch.stack(data), torch.stack(mask)

    def _next_data(self) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        n_mask_p = 1 / np.arange(1, len(self.span_info))
        n_mask_p = n_mask_p / n_mask_p.sum()
        n_masks = np.random.choice(np.arange(1, len(self.span_info)), p=n_mask_p)
        masked = np.random.choice(np.arange(len(self.span_info)), n_masks, replace=False)
        mask = torch.zeros(len(self.span_info), dtype=torch.bool, device=self.data.device)
        mask[masked] = True
        weights = self.weights[:, masked].sum(dim=1)
        selected_row = torch.multinomial(weights, num_samples=1).item()
        return self.data[selected_row].float(), mask


class InferenceDataLoader(DataLoader):
    def __init__(self,
                 n: int,
                 cache_dir: str = "diabetes-demo",
                 batch_size: int = 500):
        with open(os.path.join(cache_dir, "span-info.json"), "r") as f:
            span_info = json.load(f)
        masks = np.zeros((n, len(span_info) - 1, len(span_info)), dtype=np.bool_)
        for j in range(n):
            st = 0
            current_mask = np.zeros(len(span_info), dtype=np.bool_)
            for i in range(st, len(span_info) - 1):
                new_masked = np.random.choice(np.arange(len(span_info))[~current_mask])
                current_mask[new_masked] = True
                masks[j, i] = current_mask.copy()
        masks = torch.from_numpy(masks)
        data = torch.load(os.path.join(cache_dir, "train-data.pt"))
        indices = np.random.randint(0, data.shape[0], size=n)
        super().__init__(
            TensorDataset(data[indices].float(), masks), batch_size=batch_size
        )
