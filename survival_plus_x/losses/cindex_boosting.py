import torch


def cindex_approx(labels, risk, sigma=1.e-3, reduction_fn=torch.sum):
    """
    According to https://www.medrxiv.org/content/10.1101/2021.10.11.21264761v1.full.pdf
    section 3.4
    """
    t = labels[:, 0]
    e = labels[:, 1]

    n = len(t)

    # has the times as row vector repeated along rows
    # T2_ij = t[j]
    T2 = t.expand(n, n)

    # has the times as column vector repeated along cols
    # i.e. T1_ij = t[i]
    T1 = T2.transpose(1, 0)

    # T_ij = t[i] < t[j]
    T = (T1 < T2).long()

    # multiply by the indicator to obtain
    # E_ij = e[i]
    E = e.expand(n, n).transpose(1, 0)
    # W[i,j] = e[i] * (t[i] - t[j])
    W = E * T
    # normalization
    W = W / W.sum()

    # differences between predictions
    # P_ij = P[j] - P[i]
    # P1_ij = risk[j]
    # P2_ij = risk[i]
    P1 = risk.expand(n, n)
    P = P1 - P1.transpose(1, 0)
    # apply sigmoid
    tmp = 1. / (1 + torch.exp(P / sigma))

    # - sum_ij(w_ij * 1/(1+exp(1/sigma * (p_j - p_i))))
    return -1. * reduction_fn(W * tmp)
