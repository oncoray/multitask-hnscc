import numpy as np


def collect_from_batch_outputs(step_output_list,
                               key,
                               to_numpy=True,
                               dtype=None):

    values = []
    for step_dict in step_output_list:
        v = step_dict[key]
        if to_numpy:
            v = v.detach().cpu().numpy()

        values.extend(v)

    values = np.stack(values)

    if dtype is not None:
        values = values.astype(dtype)

    return values
