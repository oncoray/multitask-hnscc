import pandas as pd
import numpy as np
import nibabel


def transform_event_time(t, event_time_transformation):
    if event_time_transformation == "months_to_years":
        t_transform = t / 12.
    elif event_time_transformation == "log":
        t_transform = np.log(t)
    elif event_time_transformation == "identity":
        t_transform = t
    else:
        raise ValueError(
            f"Unknown event_time_transformation {event_time_transformation}")

    return t_transform


def pandas_outcome_to_dict(df,
                           id_col,
                           survival_col,
                           event_col,
                           ids=None):  # the ids to find outcome for
    outcomes = {}
    # names of columns in outcome dataframe for survival_time and event_status
    colnames = [survival_col, event_col]

    if ids is None:
        ids = df[id_col].values

    for pat_id in ids:
        # match for our id
        res = df[df[id_col] == pat_id]
        if res.empty:
            print("No outcome for patient {0}! Will skip it.".format(pat_id))
            continue
        # take only the columns that are specified
        res = res.loc[:, colnames]
        t = float(res[survival_col])
        e = int(res[event_col])

        outcomes[pat_id] = (t, e)

    return outcomes


def read_outcome(outcome_file,
                 id_col,
                 time_col,
                 event_col,
                 dropna=True,
                 csv_sep=";",
                 event_time_transformation="identity"):

    if event_time_transformation not in ['identity', 'months_to_years', 'log']:
        raise ValueError(
            f"Event_time_transformation is {event_time_transformation} but "
            f"should be one of 'identity', 'months_to_years' or 'log'!")

    outcome = pd.read_csv(outcome_file, sep=csv_sep)
    # print("\noutcome shape: {}".format(outcome.shape))

    outcome = outcome[[id_col, time_col, event_col]]

    if dropna:
        len_old = len(outcome)
        outcome = outcome.dropna()
        len_new = len(outcome)
        if len_old != len_new:
            print("Dropped {} patients due to missing outcome!".format(
                len_old - len_new))

    outcome_dict = pandas_outcome_to_dict(
        outcome,
        id_col=id_col,
        survival_col=time_col,
        event_col=event_col)

    if event_time_transformation != "identity":
        print(
            f"\n[NOTE]: Applying transformation {event_time_transformation} to the event times!")
        for pat, (t, e) in outcome_dict.items():
            t_transform = transform_event_time(t, event_time_transformation)

            outcome_dict[pat] = (t_transform, e)

    return outcome_dict


def print_event_censoring_summary(outcome_dict):
    # provide some statistics on distribution of times for each cohort
    all_times = [outcome_dict[pat][0] for pat in outcome_dict]
    event_times = [
        outcome_dict[pat][0] for pat in outcome_dict
        if outcome_dict[pat][1] == 1]
    censor_times = [
        outcome_dict[pat][0] for pat in outcome_dict
        if outcome_dict[pat][1] == 0]

    print("{} patients with outcome, {} with events ({}%)".format(
        len(all_times), len(event_times),
        np.round(100 * len(event_times) / len(all_times), 2)))

    tpl = "{} distribution: min = {}, median = {}, mean = {}, max = {}"
    names = ["all_times", "event_times", "censor_times"]
    vals = [all_times, event_times, censor_times]
    for part, time in zip(names, vals):
        if time:
            # maybe cohort without censoring or events
            print(
                tpl.format(
                    part, np.min(time),
                    np.median(time), np.mean(time), np.max(time)))


def compile_list_of_imaging_directories(input_paths):
    """
    input_paths: list of str
        full paths to the cohort directories that contain a folder for each patient.
        Folder names are assumed to be patient ids.
    """
    patient_directories = []
    for input_path in input_paths:
        pat_paths = [d for d in input_path.iterdir() if d.is_dir()]

        for pat_path in pat_paths:
            patient_directories.append(pat_path)

    return patient_directories


def load_image_as_numpy(image_path, check_shape=True):
    if image_path.suffix == ".npy":
        img_np = np.load(image_path)
    elif image_path.suffix == ".nii.gz":
        # TODO: nifti images are usually in the XYZ orientation
        # whereas numpy we have as ZYX, this should be taken care
        # of
        img_np = nibabel.load(image_path).get_fdata()
    else:
        raise ValueError(
            f"Unable to open {image_path}. Supported are .npy or .nii.gz files")

    if check_shape:
        assert img_np.ndim in [3, 4]
        if img_np.ndim == 3:
            # we prepare a channel dimension in the front if we have only
            # 3 dimensions given
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 4:
            # we have 4 dimensions as needed but we need to check if the
            # first dimension is one or if the last dimension is one
            # NOTE: this would fail for non-grayscale images but AFAIK
            # all image modalities we consider are grayscale to begin with
            if img_np.shape[-1] == 1:
                # we have channels last and need to convert to channels
                # first
                img_np = img_np.transpose(3, 0, 1, 2)
            # channels first
            assert img_np.shape[0] == 1

    return img_np


def make_surv_array_gensheimer(t, f, breaks):
    """
    Copy pasted from https://github.com/MGensheimer/nnet-survival/blob/1d728f8c9c4a5f6b886c1910bedf4cf358171dcb/nnet_survival.py#L48

    Transforms censored survival data into vector format that can be used in Keras.
      Arguments
          t: Array of failure/censoring times.
          f: Censoring indicator. 1 if failed, 0 if censored.
          breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
      Returns
          Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
    """
    assert breaks[0] == 0, "First entry must be zero!"

    n_samples = t.shape[0]
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_samples, n_intervals * 2))
    for i in range(n_samples):
        if f[i]:  # if failed (not censored)
            # give credit for surviving each time interval where failure time >= upper limit
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks[1:])
            if t[i] < breaks[-1]:  # if failure time is greater than end of last time interval, no time interval will have failure marked
                # mark failure at first bin where survival time < upper break-point
                y_train[i, n_intervals + np.where(t[i] < breaks[1:])[0][0]] = 1
        else:  # if censored
            # if censored and lived more than half-way through interval, give credit for surviving the interval.
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks_midpoint)

    return y_train
