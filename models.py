import math

import pyro
import pyro.distributions as dist
import torch


def sko_model(sko_ind_mat, lfcs=None):
    """
    Single mu for each guide.
    """

    a = pyro.sample('a', dist.Gamma(1, 1))
    b = pyro.sample('b', dist.Gamma(1, 1))

    with pyro.plate("single_kos", sko_ind_mat.shape[0]):
        mu = pyro.sample('x_i', dist.Normal(0, 1))

    mu = mu.unsqueeze(0)

    mu_e = torch.matmul(mu, sko_ind_mat)

    with pyro.plate('samples', torch.tensor(32768)):
        tau = pyro.sample('tau_s', dist.Gamma(a, b))
        tau = tau.unsqueeze(0)
        return pyro.sample('y_s', dist.Normal(mu_e, tau), obs=lfcs)


def sko_dko_model(sko_ind_mat, dko_ind_mat, lfcs=None):

    exp_noise_a = pyro.sample('a', dist.Gamma(1, 1))
    exp_noise_b = pyro.sample('b', dist.Gamma(1, 1))

    with pyro.plate('single_kos', sko_ind_mat.shape[0]):
        gene_sko_effect = pyro.sample('x_i', dist.Normal(0, 1))

    with pyro.plate('double_kos', dko_ind_mat.shape[0]):
        gene_dko_effect = pyro.sample('x_ij', dist.Normal(0, 1))

    gene_sko_effect = gene_sko_effect.unsqueeze(0)
    gene_dko_effect = gene_dko_effect.unsqueeze(0)

    mu = torch.matmul(gene_sko_effect, sko_ind_mat) \
         + torch.matmul(gene_dko_effect, dko_ind_mat)

    with pyro.plate('samples', 32768):
        tau = pyro.sample('tau_s', dist.Gamma(exp_noise_a, exp_noise_b))
        tau = tau.unsqueeze(0)
        return pyro.sample('y_s', dist.Normal(mu, tau), obs=lfcs)


def sko_tko_model(sko_ind_mat, tko_ind_mat, lfcs=None):

    exp_noise_a = pyro.sample('a', dist.Gamma(1, 1))
    exp_noise_b = pyro.sample('b', dist.Gamma(1, 1))

    with pyro.plate('single_kos', sko_ind_mat.shape[0]):
        gene_sko_effect = pyro.sample('x_i', dist.Normal(0, 1))

    with pyro.plate('triple_kos', tko_ind_mat.shape[0]):
        gene_tko_effect = pyro.sample('x_ijk', dist.Normal(0, 1))

    gene_sko_effect = gene_sko_effect.unsqueeze(0)
    gene_tko_effect = gene_tko_effect.unsqueeze(0)

    mu = torch.matmul(gene_sko_effect, sko_ind_mat) \
         + torch.matmul(gene_tko_effect, tko_ind_mat)

    with pyro.plate('samples', 32768):
        tau = pyro.sample('tau_s', dist.Gamma(exp_noise_a, exp_noise_b))
        tau = tau.unsqueeze(0)
        return pyro.sample('y_s', dist.Normal(mu, tau), obs=lfcs)


def sko_dko_tko_model(sko_ind_mat, dko_ind_mat, tko_ind_mat, lfcs=None):

    exp_noise_a = pyro.sample('a', dist.Gamma(1, 1))
    exp_noise_b = pyro.sample('b', dist.Gamma(1, 1))

    with pyro.plate('single_kos', sko_ind_mat.shape[0]):
        gene_sko_effect = pyro.sample('x_i', dist.Normal(0, 1))

    with pyro.plate('double_kos', dko_ind_mat.shape[0]):
        gene_dko_effect = pyro.sample('x_ij', dist.Normal(0, 1))

    with pyro.plate('triple_kos', tko_ind_mat.shape[0]):
        gene_tko_effect = pyro.sample('x_ijk', dist.Normal(0, 1))

    gene_sko_effect = gene_sko_effect.unsqueeze(0)
    gene_dko_effect = gene_dko_effect.unsqueeze(0)
    gene_tko_effect = gene_tko_effect.unsqueeze(0)

    mu = torch.matmul(gene_sko_effect, sko_ind_mat) \
         + torch.matmul(gene_dko_effect, dko_ind_mat) \
         + torch.matmul(gene_tko_effect, tko_ind_mat)

    with pyro.plate('samples', 32768):
        tau = pyro.sample('tau_s', dist.Gamma(exp_noise_a, exp_noise_b))
        tau = tau.unsqueeze(0)
        return pyro.sample('y_s', dist.Laplace(mu, tau), obs=lfcs)


def sko_dko_tko_guide_gene_model(sko_ind_mat,
                                 dko_ind_mat,
                                 tko_ind_mat,
                                 n_genes,
                                 lfcs=None):
    """
    Gene SKO, DKO, and TKO effects determine guide SKO, DKO, and TKO effects.
        z: gene_id
        z_i, z_j, z_k: guide effect -- guide index?
        u_i ~ N(m_z_i, t^2)
        u_i is the guide-specific effect?
        if z_i and z_j target the same gene, pull out the same gene m_z_i/j
        m ~ MVN(0, Iv^2)
    """

    exp_noise_a = pyro.sample('a', dist.Gamma(1, 1))
    exp_noise_b = pyro.sample('b', dist.Gamma(1, 1))

    sko_gene_mu = pyro.sample('g_u', dist.MultivariateNormal(torch.zeros(n_genes), covariance_matrix=torch.eye(n_genes)))
    dko_gene_mu = pyro.sample('g_uv', dist.MultivariateNormal(torch.zeros(n_genes**2), covariance_matrix=torch.eye(n_genes**2)))
    tko_gene_mu = pyro.sample('g_uvw', dist.MultivariateNormal(torch.zeros(n_genes**3), covariance_matrix=torch.eye(n_genes**3)))

    sko_gene_mu = sko_gene_mu.squeeze(0)
    dko_gene_mu = dko_gene_mu.squeeze(0)
    tko_gene_mu = tko_gene_mu.squeeze(0)

    with pyro.plate('single_kos', sko_ind_mat.shape[0]) as ind:
        sko_gene_mu_index = sko_gene_mu[torch.tensor([math.floor(i / 2) for i in ind])]
        gene_sko_effect = pyro.sample('x_i', dist.Normal(sko_gene_mu_index, 1))

    with pyro.plate('double_kos', dko_ind_mat.shape[0]) as ind:
        dko_gene_mu_index = dko_gene_mu[torch.tensor([math.floor(i / 2) for i in ind])]
        gene_dko_effect = pyro.sample('x_ij', dist.Normal(dko_gene_mu_index, 1))

    with pyro.plate('triple_kos', tko_ind_mat.shape[0]) as ind:
        tko_gene_mu_index = tko_gene_mu[torch.tensor([math.floor(i / 2) for i in ind])]
        gene_tko_effect = pyro.sample('x_ijk', dist.Normal(tko_gene_mu_index, 1))

    mu = torch.matmul(gene_sko_effect, sko_ind_mat) \
         + torch.matmul(gene_dko_effect, dko_ind_mat) \
         + torch.matmul(gene_tko_effect, tko_ind_mat)

    with pyro.plate('samples', 32768):
        tau = pyro.sample('tau_s', dist.Gamma(exp_noise_a, exp_noise_b))
        tau = tau.unsqueeze(0)
        return pyro.sample('y_s', dist.Normal(mu, tau), obs=lfcs)
