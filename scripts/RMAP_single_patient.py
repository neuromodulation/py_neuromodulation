import os
import pickle

import numpy as np
import nibabel as nib
import pandas as pd

from py_neuromodulation import nm_RMAP

if __name__ == "__main__":

    PATH_FPS = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Connectomics\DecodingToolbox_BerlinPittsburgh_Beijing\functional_connectivity"
    PATH_PERFORMANCES_BASE = (
        r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\test"
    )

    rmap_selector = nm_RMAP.RMAPChannelSelector()
    # general idea: for every patient you can do

    def get_per_for_fp(fp_names):
        fp_per = {}
        for fp_name in fp_names:
            for cohort in ["Beijing", "Berlin", "Pittsburgh"]:
                PATH_PERFORMANCES = os.path.join(
                    PATH_PERFORMANCES_BASE, f"LM_cohort_{cohort}.npy"
                )
                per = np.load(PATH_PERFORMANCES, allow_pickle=True).item()
                for sub in per.keys():
                    for run in per[sub].keys():
                        for ch in per[sub][run].keys():
                            if fp_name.startswith(f"{cohort}_{sub}_ROI_{ch}"):
                                key_name = f"{cohort}_{sub}_{ch}"
                                if key_name not in fp_per.keys():
                                    fp_per[key_name] = []
                                fp_per[key_name].append(
                                    per[sub][run][ch]["performance_test"]
                                )
        fp_per = {k: np.mean(v) for k, v in fp_per.items()}
        return fp_per

    LOAD_FPS = True

    if LOAD_FPS:
        with open(os.path.join("scripts", "fp_dict.p"), "rb") as handle:
            fp_dict = pickle.load(handle)
        with open(os.path.join("scripts", "per_all.p"), "rb") as handle:
            per_all = pickle.load(handle)
    else:
        fp_names_all, fps_all = rmap_selector.load_all_fingerprints(PATH_FPS)
        fp_dict = {k: v for k, v in zip(fp_names_all, fps_all)}
        per_all = get_per_for_fp(fp_names_all)

        with open("fp_dict.p", "wb") as handle:
            pickle.dump(fp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("per_all.p", "wb") as handle:
            pickle.dump(per_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ONE_RMAP = True
    COMPUTE_RMAP = True

    if COMPUTE_RMAP:
        fp_names_all, fps_all = rmap_selector.load_all_fingerprints(PATH_FPS)
        fp_dict = {k: v for k, v in zip(fp_names_all, fps_all)}
        per_all = get_per_for_fp(fp_names_all)
        fps_RMAP = np.array([np.nan_to_num(v.flatten()) for k, v in fp_dict.items()])

        RMAP = rmap_selector.get_RMAP(fps_RMAP.T, np.array(list(per_all.values())))

        RMAP = np.nan_to_num(RMAP)
        np.save("RMAP.npy", RMAP)

        # load example fingerprint to retrieve affine transform
        img = nib.load(
            r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Connectomics\DecodingToolbox_BerlinPittsburgh_Beijing\functional_connectivity\Beijing_FOG006_ROI_ECOG_R_1_SM_HH-avgref_func_seed_AvgR_Fz.nii"
        )

        rmap_selector.save_Nii(RMAP, img.affine, "RMAP.nii", reshape=True)

    RMAP = np.load(os.path.join("scripts", "RMAP.npy"))

    ch_sel = {}
    for cohort in ["Beijing", "Berlin", "Pittsburgh"]:
        PATH_PERFORMANCES = os.path.join(
            PATH_PERFORMANCES_BASE, f"LM_cohort_{cohort}.npy"
        )
        per = np.load(PATH_PERFORMANCES, allow_pickle=True).item()
        print(cohort)
        for sub in per.keys():
            print(sub)
            str_test = f"{cohort}_{sub}"

            per_filter_RMAP = {k: v for k, v in per_all.items() if str_test not in k}
            fps_RMAP = [
                np.nan_to_num(v) for k, v in fp_dict.items() if str_test not in k
            ]

            if ONE_RMAP is False:
                RMAP = rmap_selector.calculate_RMap(
                    fps_RMAP, np.nan_to_num(list(per_filter_RMAP.values()))
                )

                RMAP = np.nan_to_num(RMAP)

            for rec in per[sub].keys():
                # get first ECoG channel from recording to check if sess is left or right
                sess_ch = [k for k in list(per[sub][rec].keys()) if "ECOG" in k][0]
                if "_LEFT_" in sess_ch or "_L_" in sess_ch:
                    per_filter_test = {
                        k: v
                        for k, v in per_all.items()
                        if str_test in k and ("_LEFT_" in k or "_L_" in k)
                    }
                    fps_dict_test = {
                        k: v
                        for k, v in fp_dict.items()
                        if str_test in k and ("_LEFT_" in k or "_L_" in k)
                    }
                else:
                    per_filter_test = {
                        k: v
                        for k, v in per_all.items()
                        if str_test in k and ("_RIGHT_" in k or "_R_" in k)
                    }
                    fps_dict_test = {
                        k: v
                        for k, v in fp_dict.items()
                        if str_test in k and ("_RIGHT_" in k or "_R_" in k)
                    }
                chs = []
                for ch_name_test, fp_test in fps_dict_test.items():
                    corr_ = np.nan_to_num(
                        rmap_selector.get_corr_numba(fp_test.flatten(), RMAP)
                    )
                    # get ch
                    chs.append([ch_name_test, corr_])
                max_corr_ch = np.argmax(np.array(chs)[:, 1])
                max_per = chs[max_corr_ch]

                ch_best = chs[max_corr_ch][0][
                    chs[max_corr_ch][0].find("ROI_")
                    + 4 : chs[max_corr_ch][0].find("_func_seed")
                ]

                if cohort not in ch_sel.keys():
                    ch_sel[cohort] = {}
                if sub not in ch_sel[cohort].keys():
                    ch_sel[cohort][sub] = {}

                if rec not in ch_sel[cohort][sub].keys():
                    ch_sel[cohort][sub][rec] = {}
                if ch_best in per[sub][rec].keys():
                    ch_sel[cohort][sub][rec] = per[sub][rec][ch_best]
                    ch_sel[cohort][sub][rec]["max_corr_ch"] = ch_best
                    ch_sel[cohort][sub][rec]["max_corr"] = max_per[1]
                else:
                    print("ch best not in per")
                    print(f"ch: {ch_best} sub: {sub} cohort: {cohort} rec: {rec}")

    with open("ch_sel.p", "wb") as handle:
        pickle.dump(ch_sel, handle, protocol=pickle.HIGHEST_PROTOCOL)
