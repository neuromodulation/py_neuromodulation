# py_neuromodulation

The py_neuromodulation module allows for real time capable processing of multimodal electrophysiological data. The primary use is movement prediction for [adaptive deep brain stimulation](https://pubmed.ncbi.nlm.nih.gov/30607748/).

For running this toolbox first create a new virtual conda environment:

`conda create -n pyneuromodulation python=3.9`

Then activate it:

`conda activate pyneuromodulation`

and install the required packages:

`pip install -r requirements.txt --user`

For running the LabStreamingLayer example mne-realtime also needs to be installed:

`pip install https://api.github.com/repos/mne-tools/mne-realtime/zipball/master`

The main modules include running real time enabled feature preprocessing based on [iEEG BIDS](https://www.nature.com/articles/s41597-019-0105-7) data. Different features can be enabled/disabled and parametrized in the [settings.json](https://github.com/neuromodulation/py_neuromodulation/blob/main/examples/settings.json). The current implementation mainly focuses band power and [sharpwave](https://www.sciencedirect.com/science/article/abs/pii/S1364661316302182) feature estimation.

An example folder with a mock subject and derivate [feature](https://github.com/neuromodulation/py_neuromodulation/tree/main/pyneuromodulation/tests/data/derivatives/sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg) set was estimated.
