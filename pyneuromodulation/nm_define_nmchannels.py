from numpy import arange, nan
from pandas import DataFrame


def set_nm_channels(ch_names, ch_types, reference='default', bads=None,
                    new_names='default', ECOG_ONLY=False):
    """Return dataframe with channel-specific settings.

    Return an nm_channels dataframe with the columns: "name", "rereference", "used",
    "target", "type"]. "name" is set to ch_names, "rereference" can be
    specified individually. "used" is set to 1 for all channels containing
    "ECOG", "SEEG", "LFP" or "DBS", else to 0. "target" is set to 1 for all
    channels containing "TTL", "ANALOG", "MOV" and "ROTA", else to 0.

    Possible channel types:
    https://github.com/mne-tools/mne-python/blob/6ae3b22033c745cce5cd5de9b92da54c13c36484/doc/_includes/channel_types.rst

    Arguments
    ---------
        ch_names : list
            list of channel names. Should each contain one of the keywords
            (lower or upper case): "ECOG", "LFP", "DBS", "SEEG", "TTL",
            "ANALOG", "ROTA" or "MOV".
        ch_types : list
            list of channel types. Should optimally be of the types: "ECOG",
            "DBS" or "SEEG". "LFP" is also accepted.
        reference : str | list of str | None, default: 'default'
            re-referencing scheme. Default is "default". This sets ECOG channel
            references to "average" and creates a bipolar referencing scheme
            for LFP/DBS/SEEG channels, where each channel is referenced to
            the adjacent lower channel, split by left and right hemisphere.
            For this, the channel names must contain the substring "_L_" and/or
            "_R_" (lower or upper case). CAVE: Adjacent channels will be
            determined using the sorted() function.
        bads :  str | list of str, default: None
            channels that should be marked as bad and not be used for
            average re-referencing etc.
        new_names : list of str | None, default: 'default'
            new channel names that should be used when writing out the
            features and results. Useful when applying re-referencing. Set to
            `None` if no renaming should be performed. 'default' will infer
            channel renaming from re-referencing information. If a list is
            given, it should be in the same order as `ch_names`.
        ECOG_ONLY : boolean, default: False
            if True, set only 'ecog' channel type to used

    Returns
    -------
        df: DataFrame in nm_channels format
    """

    if not (len(ch_names) == len(ch_types)):
        raise Exception("Sorry, no numbers below zero")

    mov_substrs = ['analog', 'mov', 'rota_squared', 'squared_emg', 'squared_rotawheel']
    used_substrs = ['ecog', 'seeg']
    lfp_types = ['seeg', 'dbs', 'lfp']

    df = DataFrame(nan, index=arange(len(list(ch_names))),
                   columns=['name', 'rereference', 'used', 'target', 'type',
                            'status', 'new_name'])
    df['name'] = ch_names

    df['used'] = [1 if any(used_substr.lower() in ch_name.lower()
                           or used_substr.lower() in ch_type.lower()
                           for used_substr in used_substrs)
                  else 0 for ch_type, ch_name in zip(ch_types, ch_names)]

    df['target'] = [1 if any(mov_substr.lower() in ch_name.lower()
                             for mov_substr in mov_substrs) else 0
                    for ch_idx, ch_name in enumerate(ch_names)]

    # note: BIDS types are in caps, mne.io.RawArray types lower case
    # s.t. 'type' will be in lower case here
    df['type'] = ch_types

    if ECOG_ONLY is True:
        df.loc[(df["type"] == "seeg") | (df["type"] == "dbs"), "used"] = 0

    if isinstance(reference, str):
        if not (reference in ['default', 'Default']):
            raise ValueError("`reference` must be either `default`, None or "
                             "an iterable of new reference channel names. "
                             f"Got: {reference}.")
        ecog_chs = []
        lfp_chs = []
        other_chs = []
        for ch_name, ch_type in zip(ch_names, ch_types):
            if "ecog" in ch_type.lower() or "ecog" in ch_name.lower():
                ecog_chs.append(ch_name)
            elif any(lfp_type.lower() in ch_type.lower()
                     or lfp_type.lower() in ch_name.lower()
                     for lfp_type in lfp_types):
                lfp_chs.append(ch_name)
            else:
                other_chs.append(ch_name)
        lfp_l = sorted(
            [lfp_ch for lfp_ch in lfp_chs if ('_l_' in lfp_ch.lower())
             or ('_left_' in lfp_ch.lower())])
        lfp_r = sorted(
            [lfp_ch for lfp_ch in lfp_chs if ('_r_' in lfp_ch.lower())
             or ('_right_' in lfp_ch.lower())])
        lfp_l_refs = [lfp_l[i - 1] if i > 0 else lfp_l[-1] for i in
                      range(len(lfp_l))]
        lfp_r_refs = [lfp_r[i - 1] if i > 0 else lfp_r[-1] for i in
                      range(len(lfp_r))]
        ref_idx = list(df.columns).index('rereference')
        for ecog_ch in ecog_chs:
            df.iloc[
                df[df['name'] == ecog_ch].index[0], ref_idx] = 'average'
        for i, lfp in enumerate(lfp_l):
            df.iloc[
                df[df['name'] == lfp].index[0], ref_idx] = lfp_l_refs[i]
        for i, lfp in enumerate(lfp_r):
            df.iloc[
                df[df['name'] == lfp].index[0], ref_idx] = lfp_r_refs[i]
        for other_ch in other_chs:
            df.iloc[df[df['name'] == other_ch].index[0], ref_idx] = 'None'
    elif hasattr(reference, '__iter__'):
        if len(reference) != len(ch_names):
            raise ValueError(
                "`new_names` must have the same number of items as the "
                "original channel names.")
        df['reference'] = reference
    elif not reference:
        df.loc[:, 'rereference'] = 'None'
    else:
        raise ValueError("`reference` must be either `default`, None or "
                         "an iterable of new reference channel names. "
                         f"Got: {reference}.")

    if bads is not None:
        if isinstance(bads, str):
            bads = [bads]
        df['status'] = ['bad' if ch in bads else 'good' for ch in ch_names]
        df.loc[df['status'] == 'bad', 'used'] = 0  # setting bad channels to not being used
    else:
        df['status'] = 'good'

    if not new_names:
        df['new_name'] = ch_names
    elif isinstance(new_names, str):
        if new_names.lower() != 'default':
            raise ValueError("`new_names` must be either `default`, None or "
                             "an iterable of new channel names. Got: "
                             f"{new_names}.")
        new_names = list()
        for name, ref in zip(df['name'], df['rereference']):
            if ref == 'None':
                new_names.append(name)
            elif ref == 'average':
                new_names.append(name + '-avgref')
            else:
                new_names.append(name + '-' + ref)
        df['new_name'] = new_names
    elif hasattr(new_names, '__iter__'):
        if len(new_names) != len(ch_names):
            raise ValueError("`new_names` must have the same number of items"
                             "as the original channel names.")
        else:
            df['new_name'] = ch_names
    else:
        raise ValueError("`new_names` must be either `default`, None or "
                         "an iterable of new channel names.Got: "
                         f"{new_names}.")
    return df
