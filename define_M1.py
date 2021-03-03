import pandas as pd
import numpy as np

mov_substr_list = ["TTL", "ANALOG", "MOV"]
used_substr_list = ["ecog", "seeg"]

def set_M1(ch_names, ch_type):
    """
    Return a default M1 dataframe that set's 
    "name" to parameter ch_names 
    "rereference" to "average" 
    "used" for all "ECOG" and "SEEG" to 1 
    "target" for all "TTL", "ANALOG" and "MOV" to 1 

    Args:
        ch_names (list): channel names 
        ch_type (list): channel types
    
    Returns: 
        df in M1 format
    """

    # set here: name, reference, used, target, ECoG 
    df = pd.DataFrame(np.nan, index=np.arange(len(list(ch_names))), columns=['name', 'rereference', 'used', 'target'])

    ch_used = [ch_idx for ch_idx, ch in enumerate(ch_type) if any(used_substr in ch for used_substr in used_substr_list)]
    used = np.zeros(len(ch_names))
    used[ch_used] = 1
    df['used'] = used.astype(int)
    df['name'] = ch_names


    # check here for TTL, ANALOG 
    ch_mov = [ch_idx for ch_idx, ch in enumerate(ch_names) if any(mov_substr in ch for mov_substr in mov_substr_list)]
    target = np.zeros(len(ch_names))
    target[ch_mov] = 1
    df['target'] = target.astype(int)

    df['rereference'] = ["average" for x in range(len(ch_names))]

    return df