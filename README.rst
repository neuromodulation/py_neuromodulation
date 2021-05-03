py_neuromodulation
==================

The py_neuromodulation toolbox allows for real time capable processing of multimodal electrophysiological data. The primary use is movement prediction for `adaptive deep brain stimulation <https://pubmed.ncbi.nlm.nih.gov/30607748/>`_.

Find the documentation here https://neuromodulation.github.io/py_neuromodulation/ for example usage and parametrization. 

For running this toolbox first create a new virtual conda environment:

.. code-block::

    conda create -n pyneuromodulation python=3.9

Then activate it:

.. code-block::

    conda activate pyneuromodulation

and install the required packages:

.. code-block::

    pip install -r requirements.txt --user

For running the LabStreamingLayer example mne-realtime also needs to be installed:

.. code-block::

    pip install https://api.github.com/repos/mne-tools/mne-realtime/zipball/master

The main modules include running real time enabled feature preprocessing based on `iEEG BIDS <https://www.nature.com/articles/s41597-019-0105-7>`_ data. 

Different features can be enabled/disabled and parametrized in the `settings.json <https://github.com/neuromodulation/py_neuromodulation/blob/main/examples/settings.json>`_. 

The current implementation mainly focuses band power and `sharpwave <https://www.sciencedirect.com/science/article/abs/pii/S1364661316302182>`_ feature estimation.

An example folder with a mock subject and derivate `feature <https://github.com/neuromodulation/py_neuromodulation/tree/main/pyneuromodulation/tests/data/derivatives/sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg>`_ set was estimated.

To run feature estimation given the example BIDS data run in the example folder. 

.. code-block:: 

    python example_BIDS.py

This will write write a feature_arr.csv file in the 'pyneuromodulation/tests/data/derivatives' folder. 
To estimate next a basic machine learning pipeline including Bayesian Optimization for xgboost, run 
.. code-block::

    python example_ML.py

This will save an 'ML_res.p' file in the 'pyneuromodulation/tests/data/derivatives' folder.

Next the features can be investigated using 
.. code-block::

    python example_read_features.py

In such manner features are plotted and individual channels are visualized on the MNI brain. 

For further documentatin view `ParametrizationDefinition <ParametrizationDefinition.html#>`_ for description of necessary parametrization files. 
`FeatureEstimationDemo <FeatureEstimationDemo.html#>`_ walks through an example feature estimation and explains sharpwave estimation. 
Note, to install an ipython kernel for the upper installed environment, install jupyter lab and  a respective ipython kernel:
.. code-block::

    conda install -c conda-forge jupyterlab    
    ipython kernel install --user --name=pyneuromodulation