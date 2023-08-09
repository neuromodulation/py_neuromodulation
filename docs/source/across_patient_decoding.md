## Across patient decoding using R-Map optimal connectivity

ECoG electrode placement is commonly very heretogeneous across patients and cohorts. To still facilitate approaches that are able to perform decoding applications without patient individual training, two across-patient decoding approaches were previously investigated for movement decoding:
- grid-point decoding
- optimal connectivity channel decoding

First, the grid-point decoding approach relies on definition of a cortical or subcortical grid. Data from individual grid points is then interpolated onto those common grid points. The approach was also explained in the example_gridPointProjection.ipynb notebook.

The R-Map decoding approach relies on the other hand on computation of whole brain connectivity. Therefore, the electrode MNI space locations need to be known, the following steps can be the performed for decoding without patient individual training:

1. The electrode localizations in MNI space need to be known. Using the [wjn_toolbox](https://github.com/neuromodulation/wjn_toolbox) function
    ```
    wjn_spherical_roi(roiname,mni,4)
    ```
    function, sphecrical NIFTI (.nii) files can be computed that coontain the electrode contact region of interest.
2. For the given ROI.nii files, the LeadDBS [LeadMapper](https://netstim.gitbook.io/leaddbs/connectomics/lead-mapper) tool can be used for functional or structual connectivity estimation. 
3. The py_neuromodulation *nm_RMAP.py* module can then compute the R-Map given the contact-individual connectivity fingerprints. 
    ```
    nm_RMAP.calculate_RMap_numba(fingerprints, performances)
    ```
4. The fingerprints from test-set patients can then be correlated with the calculated R-Map:
   ```
   nm_RMAP.get_corr_numba(fp, fp_test)
   ```
5. The channel with highest correlation can then be selected for decoding without individual training. *nm_RMAP* contains aleady leave one channel and leave one patient out cross validation functions  
   ```
   nm_RMAP.leave_one_sub_out_cv(l_fps_names, l_fps_dat, l_per, sub_list)
   ```
6. The obtained R-Map correlations can then be estimated statistically, and plooted agains true correlates:
   ```
   nm_RMAP.plot_performance_prediction_correlation(per_left_out, per_predict, out_path_save)
   ```
