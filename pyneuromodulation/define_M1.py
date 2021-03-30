from numpy import arange, nan
from pandas import DataFrame


def set_M1(ch_names, ch_types, reference='default'):
    """Return dataframe with channel-specific settings.

    Return an M1 dataframe with the columns: "name", "rereference", "used",
    "target", "ECOG"]. "name" is set to ch_names, "rereference" can be specified
    individually. "used" is set to 1 for all channels containing "ECOG", "SEEG",
    "LFP" or "DBS", else to 0. "target" is set to 1 for all channels containing
    "TTL", "ANALOG", "MOV" and "ROTA", else to 0

    Args:
        ch_names (list):
            list of channel names. Should each contain one of the keywords
            (lower or upper case): "ECOG", "LFP", "DBS", "SEEG", "TTL",
            "ANALOG", "ROTA" or "MOV".
        ch_types (list):
            list of channel types. Should optimally be of the types: "ECOG",
            "DBS" or "SEEG". "LFP" is also accepted.
        reference (string or list of strings):
            re-referencing scheme. Default is "default". This sets ECOG channel
            references to "average" and creates a bipolar referencing scheme
            for LFP/DBS/SEEG channels, where each channel is referenced to
            the adjacent lower channel, split by left and right hemisphere.
            For this, the channel names must contain the substring "_L_" and/or
            "_R_" (lower or upper case). CAVE: Adjacent channels will be
            determined using the sorted() function.

    
    Returns: 
        df: dataframe in M1 format
    """

    if not (len(ch_names) == len(ch_types)):
        raise Exception("Sorry, no numbers below zero")

    mov_substrs = ["ttl", "analog", "mov", "rota"]
    used_substrs = ["ecog", "seeg"]
    lfp_types = ['seeg', 'dbs', 'lfp']

    df = DataFrame(nan, index=arange(len(list(ch_names))),
                      columns=['name', 'rereference', 'used', 'target', 'ECOG'])
    df['name'] = ch_names

    df['used'] = [1 if any(used_substr.lower() in ch_name.lower()
                           or used_substr.lower() in ch_type.lower()
                           for used_substr in used_substrs)
                  else 0 for ch_type, ch_name in zip(ch_types, ch_names)]

    df['target'] = [1 if any(mov_substr.lower() in ch_name.lower()
                             for mov_substr in mov_substrs) else 0
                    for ch_idx, ch_name in enumerate(ch_names)]

    df['ECOG'] = [1 if 'ecog' in ch_name.lower() or 'ecog'
                  in ch_types[ch_idx].lower()
                  else 0 for ch_idx, ch_name in enumerate(ch_names)]

    if any(reference):
        if reference in ['default', 'Default']:
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
                [lfp_ch for lfp_ch in lfp_chs if '_l_' in lfp_ch.lower()])
            lfp_r = sorted(
                [lfp_ch for lfp_ch in lfp_chs if '_r_' in lfp_ch.lower()])
            lfp_l_refs = [lfp_l[i - 1] if i > 0 else 'None' for i in
                          range(len(lfp_l))]
            lfp_r_refs = [lfp_r[i - 1] if i > 0 else 'None' for i in
                          range(len(lfp_r))]
            ref_idx = list(df.columns).index('rereference')
            for ecog_ch in ecog_chs:
                df.iloc[df[df['name'] == ecog_ch].index[0], ref_idx] = 'average'
            for i, lfp in enumerate(lfp_l):
                df.iloc[df[df['name'] == lfp].index[0], ref_idx] = lfp_l_refs[i]
            for i, lfp in enumerate(lfp_r):
                df.iloc[df[df['name'] == lfp].index[0], ref_idx] = lfp_r_refs[i]
            for other_ch in other_chs:
                df.iloc[df[df['name'] == other_ch].index[0], ref_idx] = 'None'
        else:
            df['rereference'] = reference
    else:
        df.loc[:, 'rereference'] = "None"
    return df
