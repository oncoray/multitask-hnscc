import torch


def cox_log_likelihood(labels, risk,
                       reduction_fn=torch.sum,
                       average_over_events_only=True):
    """
    labels: torch.tensor (B x 2)
        First column have to be event times, second column the binary event
        indicators where 1 specifies occurence of event and 0 censoring
    risk: torch.tensor (B,)
        The predicted risk scores
    reduction_fn: either torch.sum or torch.mean
    average_over_events_only: bool
        only used if reduction_fn==torch.mean and means we divide the sum by the
        number of samples with events, not the number of samples in the batch
    """
    assert reduction_fn in [torch.sum, torch.mean]

    t = labels[:, 0]
    e = labels[:, 1]

    # sort times decreasing
    idx = torch.argsort(t, descending=True)

    e_ordered = e[idx]
    risk_ordered = risk[idx]

    logsumexp = torch.log(torch.cumsum(torch.exp(risk_ordered), 0))
    result_per_patient = risk_ordered - logsumexp

    result = torch.sum(e_ordered * result_per_patient)

    if reduction_fn == torch.mean:
        if average_over_events_only:
            n_events = e_ordered.sum()
            if n_events > 0:
                result /= n_events
            else:
                assert result == 0
        else:
            # we average over all samples in the batch
            result /= e_ordered.shape[0]

    return result


def neg_cox_log_likelihood(labels, risk,
                           reduction_fn=torch.sum,
                           average_over_events_only=False):
    # NOTE: default for average_over_events_only is set to False to
    # not break current behaviour in the loss functions used!
    return -1 * cox_log_likelihood(
        labels, risk, reduction_fn, average_over_events_only)


def cox_log_likelihood_with_decay(labels, risk, decays,
                                  reduction_fn=torch.sum,
                                  average_over_events_only=True):
    """
    When using memory banks we might want to downweight contributions
    of patients whose predictions have not been updated recently (i.e. come
    from older models).
    """
    assert reduction_fn in [torch.sum, torch.mean]

    t = labels[:, 0]
    e = labels[:, 1]

    # sort times decreasing, e.g. 10, 8, 5, 2
    idx = torch.argsort(t, descending=True)

    e_ordered = e[idx]
    risk_ordered = risk[idx]
    decays_ordered = decays[idx]

    logsumexp = torch.log(torch.cumsum(
        decays_ordered * torch.exp(risk_ordered), 0))
    result_per_patient = risk_ordered - logsumexp

    result = torch.sum(e_ordered * result_per_patient)

    if reduction_fn == torch.mean:
        if average_over_events_only:
            n_events = e_ordered.sum()
            if n_events > 0:
                result /= n_events
            else:
                assert result == 0
        else:
            # we average over all samples in the batch
            result /= e_ordered.shape[0]

    return result


def neg_cox_log_likelihood_with_decay(labels, risk, decays,
                                      reduction_fn=torch.sum,
                                      average_over_events_only=False):
    return -1 * cox_log_likelihood_with_decay(
        labels, risk, decays, reduction_fn, average_over_events_only)
