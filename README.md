# clean_edf

This package removes personal identifiers (i.e., de-identifies) clinical EEG collected on hospital recording systems. It was originally developed by Riley DeHaan at the Computational Memory Lab at the University of Pennsylvania to facilitate collecting, analyzing, and publishing intracranial EEG. This code currently supports de-identification of EEG recordings stored in the [European Data Format](https://www.edfplus.info/) (EDF) standard. This package was developed primarily using Nihon Kohden (NK) recording exports to EDF (which break the EDF standard in certain ways; see below). We expect other clinical systems would introduce different quirks into EDF exports that may necessitate updating this package. 

De-identification operations:
- Replace patient names in the EDF header and optionally in EDF annotations (we find that technicians sometimes place patient names in the annotations, which can contain arbitrary text and should not be assumed to be free of identifying information).
- Drop annotations containing other identifying information, including gendered pronouns and arbitrary regex patterns
- Set recording start times to 1985-01-01 (with relative offsets from the time of the first recording if multiple EDF files from the same subject are processed together to preserve relative timing information)

Installation:
```
git clone git@github.com:Riley16/clean_eeg.git
cd clean_eeg
# create new conda environment
conda create -n clean_eeg python=3.11
conda activate clean_eeg
pip install .
```

To run tests (from a current working directory of the clean_eeg/ install directory), instead install in editable mode:
```
conda activate clean_eeg
pip install -e .
python tests/generate_edf.py
pytest
```

To clean a directory of EEG files from a given subject:
```
cd clean_eeg
conda activate clean_eeg
python src/clean_eeg/clean_subject_eeg.py --input_path PATH/TO/ALL/SUBJECT/EEG/FILES --output_path OPTIONAL/PATH/TO/OUTPUT/DEIDENTIFIED/FILES --subject_code R1XXXY --first-name John --middle-name Paul --last-name Smith
```
The path to the de-identified files will be printed once the process finishes.

Dependencies:
- [pyedflib](https://github.com/holgern/pyedflib)
    - our primary EDF IO manager
- [lunapi](https://zzz.bwh.harvard.edu/luna/lunapi/)
    - used to split discontinuous EDF+D files into separate continuous EDF+C files since pyedflib does not support the EDF+D formats
- [edfio](https://github.com/the-siesta-group/edfio)
- [MNE](https://mne.tools/stable/index.html)

Issues:
- NK recording systems export EDF files that break the EDF standard. For instance, NK will often include a partial recording block (i.e., with fewer samples than the other recording blocks) for the last recording block in a file. The different python libraries used in this package deal with partial blocks in different ways to comply with the EDF standard. The lunapi library silently drops partial recording blocks, representing a small amount of data loss. The pyedflib library zero pads partial blocks. Zero padding would be better, particularly if an annotation indicating the start of zero padding were added. However, pyedflib does not support the EDF+D files exported by NK. For now, we accept the small amount of data loss to use lunapi.
