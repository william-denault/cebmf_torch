"""Independently integrate marginal mixture KLs to audit the ELBO identity."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from cebmf_torch.cebnm.cov_gb_prior import cgb_posterior_means
from cebmf_torch.cebnm.cov_sharp_2gb_prior import sharp_2cgb_posterior_means
from cebmf_torch.cebnm.spiked_emdn import spiked_emdn_posterior_means
from cebmf_torch.ebnm.ash import ash


def direct_kl(x, se, pi, mu, sd, points):
    """Separate the exact spike mass from continuous mixture densities."""
    x, se, pi, mu, sd = [v.double() for v in (x, se, pi, mu, sd)]
    n = len(x)
    pi, mu, sd = [v.expand(n, -1) for v in (pi, mu, sd)]
    s2, t2 = se[:, None].square(), sd.square()
    log_evidence_component = -.5 * (math.log(2 * math.pi) + (s2 + t2).log()
                                    + (x[:, None] - mu).square() / (s2 + t2))
    joint = pi.log() + log_evidence_component
    log_evidence = joint.logsumexp(1)
    log_r = joint - log_evidence[:, None]
    r = log_r.exp()
    m = (t2 * x[:, None] + s2 * mu) / (s2 + t2)
    v = s2 * t2 / (s2 + t2)
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    nodes = torch.from_numpy(nodes) * math.sqrt(2)
    weights = torch.from_numpy(weights) / math.sqrt(math.pi)
    values = m[:, 1:, None] + v[:, 1:, None].sqrt() * nodes

    def log_continuous(log_weight, center, variance):
        log_density = -.5 * (math.log(2 * math.pi) + variance[:, None, None, :].log()
                            + (values[..., None] - center[:, None, None, :]).square()
                            / variance[:, None, None, :])
        return (log_density + log_weight[:, None, None, :]).logsumexp(-1)

    log_q = log_continuous(log_r[:, 1:], m[:, 1:], v[:, 1:])
    log_p = log_continuous(pi[:, 1:].log(), mu[:, 1:], t2[:, 1:])
    kl = r[:, 0] * (log_r[:, 0] - pi[:, 0].log())
    kl += (r[:, 1:] * ((log_q - log_p) * weights).sum(-1)).sum(1)
    first = (r * m).sum(1)
    second = (r * (v + m.square())).sum(1)
    return kl.sum(), first, second, log_evidence.sum()


def run():
    torch.set_num_threads(1)
    rows = []
    for moment_mode in ('mean', 'mean_second'):
        torch.manual_seed(423)
        parents = torch.randn(12, 2)
        second = parents.square() + torch.rand(12, 2)
        inputs = parents if moment_mode == 'mean' else torch.cat((parents, second), 1)
        x = torch.linspace(-1.7, 2.3, 12)
        se = torch.linspace(.25, .7, 12)
        for name, fitter in [('cgb', cgb_posterior_means), ('cgb_sharp_2', sharp_2cgb_posterior_means),
                             ('spiked_emdn', spiked_emdn_posterior_means), ('ash', None)]:
            torch.manual_seed(97)
            if name == 'ash':
                fit = ash(x, se, prior='norm', scales=[0., .1, .5, 2.], penalty=10.)
                pi, sd = fit.pi[None, :], fit.scale[None, :]
                mu = torch.zeros_like(sd)
                loss = -fit.log_lik
            else:
                fit = fitter(inputs, x, se, n_epochs=2, hidden_dim=5, n_layers=0,
                             batch_size=12, penalty=1.1, device='cpu')
                loss = fit.loss
                zeros = torch.zeros_like(x)
                if name == 'cgb':
                    pi = torch.stack((fit.pi, 1 - fit.pi), 1)
                    mu = torch.stack((zeros, fit.mu_2.expand_as(x)), 1)
                    sd = torch.stack((zeros, fit.sigma_2.expand_as(x)), 1)
                elif name == 'cgb_sharp_2':
                    pi = torch.stack((fit.pi, fit.pi_pos, fit.pi_neg), 1)
                    mu = torch.stack((zeros, fit.mu_pos.expand_as(x), fit.mu_neg.expand_as(x)), 1)
                    sd = torch.stack((zeros, fit.sigma_1.expand_as(x), fit.sigma_2.expand_as(x)), 1)
                else:
                    pi, mu, sd = fit.pi_np, fit.location, fit.scale
            direct, m, m2, log_evidence = direct_kl(x, se, pi, mu, sd, 128)
            refined = direct_kl(x, se, pi, mu, sd, 256)[0]
            nm_ll = -.5 * (math.log(2 * math.pi) + 2 * se.double().log()
                          + (x.double().square() - 2 * x.double() * fit.post_mean.double()
                             + fit.post_mean2.double()) / se.double().square()).sum()
            identity = nm_ll + loss.double()
            torch.testing.assert_close(m, fit.post_mean.double(), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(m2, fit.post_mean2.double(), rtol=2e-5, atol=2e-6)
            assert abs(float(identity - refined)) < 2e-4
            assert abs(float(direct - refined)) < 1e-6
            rows.append(dict(method=moment_mode, prior=name, direct_marginal_kl=float(refined),
                             evidence_identity_kl=float(identity), absolute_error=float(abs(identity - refined)),
                             quadrature_refinement_error=float(abs(direct - refined)),
                             marginal_loglik_error=float(abs(log_evidence + loss.double()))))
    folder = Path('output/moment_covariates_20260917')
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(passed=True, checks=rows, note='Independent marginal-mixture entropy/KL integration, including exact point-mass contribution; CPU only.')
    (folder / 'elbo_identity_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
