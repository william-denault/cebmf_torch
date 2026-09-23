"""Benchmark-only controls; these do not add fitting modes to the public API."""

import torch

from cebmf_torch.cebmf._conditional import LoadingGraph


class AblationGraph(LoadingGraph):
    parents_at_mean = False
    feedback_enabled = True

    def integrated_inputs(self, u, index, draws, excluding=None):
        if not self.parents_at_mean:
            return super().integrated_inputs(u, index, draws, excluding)
        context = self.reference_inputs(u)[index, None, :].clone()
        if excluding is not None:
            position = self.external[u].shape[1] + self.parents[u].index(excluding)
            context[:, :, position].zero_()
        return context, context.new_ones(len(index), 1)

    def child_term(self, u, index, values, draws, prior_states=None):
        if not self.feedback_enabled:
            return torch.zeros_like(values)
        return super().child_term(u, index, values, draws, prior_states)
