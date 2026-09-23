"""Batch density-regression tensors without CPU index generation on CUDA."""

import math

import torch
from torch.utils.data import DataLoader


class _DeviceBatches:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = len(dataset) if batch_size is None else batch_size
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if not self.shuffle:
            for start in range(0, n, self.batch_size):
                yield self.dataset[slice(start, start + self.batch_size)]
        else:
            indices = torch.randperm(n, device=self.dataset.X.device)
            for batch in indices.split(self.batch_size):
                yield self.dataset[batch]


def density_batches(dataset, batch_size, shuffle=True):
    # Preserve the existing CPU shuffle/RNG schedule for ordinary cEBNM.
    if not dataset.X.is_cuda:
        return DataLoader(dataset, batch_size=len(dataset) if batch_size is None else batch_size,
                          shuffle=shuffle, num_workers=0)
    return _DeviceBatches(dataset, batch_size, shuffle)
