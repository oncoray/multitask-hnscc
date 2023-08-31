import torch


def neg_log_likelihood(labels, pred, reduction_fn=torch.sum):
    """
    Arguments
        labels: Tensor.
          First half of the values is 1 if individual survived that interval,
          0 if not.
          Second half of the values is for individuals who failed,
          and is 1 for time interval during which failure occured,
          0 for other intervals.
          See make_surv_array function.
        pred: Tensor, predicted survival probability (1-hazard probability)
              for each time interval.
        reduction_fn: torch function
            how to aggregate the individual likelihoods for the minibatch
            (e.g. mean or sum)
    Returns
        sum or mean over vector of log likelihoods for this minibatch.
    """
    # labels has to be a matrix with even number of columns
    n_intervals = labels.shape[1] // 2
    assert 2 * n_intervals == labels.shape[1]
    epsilon = 1.e-10

    # component for all individuals
    cens_uncens = 1. + labels[:, 0:n_intervals] * (pred - 1.)
    # component for only uncensored individuals
    uncens = 1. - labels[:, n_intervals:2*n_intervals] * pred
    concat = torch.cat((cens_uncens, uncens), dim=-1)
    clipped = torch.clip(concat, min=epsilon, max=None)
    logs = clipped.log()

    loglik_per_patient = logs.sum(dim=-1)

    return -1. * reduction_fn(loglik_per_patient)
