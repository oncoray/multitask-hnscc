import torch


def log_likelihood(labels,
                   predicted_distributions,
                   reduction_fn=torch.mean):
    """
    Computes the logarithm of the general survival likelihood
    for right censored data:
        L(t,e) = S(t)^(1-e) * f(t)^e
    for a batch of predicted distribution objects.
    This can be formulated as
        logL = sum( (1-e_i)*logS(t_i) + e_i * logf(t_i))
    """

    t = labels[:, 0]
    e = labels[:, 1]

    log_f = predicted_distributions.log_prob(t)
    # NOTE: this might cause numerical problems (infty or nan) if t is large
    # and the distribution parameters havent been adjusted properly.
    #log_S = torch.log(1. - predicted_distributions.cdf(t))
    log_S = torch.log1p(-predicted_distributions.cdf(t))

    log_L = reduction_fn(e*log_f + (1-e)*log_S)

    return log_L


def neg_log_likelihood(labels,
                       predicted_distributions,
                       reduction_fn=torch.mean):

    return -1. * log_likelihood(
        labels, predicted_distributions, reduction_fn)
