# clean_edf

This package removes personal identifiers (i.e., de-identifies) clinical EEG collected on hospital recording systems. It was originally developed by Riley DeHaan at the Computational Memory Lab at the University of Pennsylvania to facilitate collecting, analyzing, and publishing intracranial EEG. This code currently supports de-identification of EEG recordings stored in the [European Data Format](https://www.edfplus.info/) (EDF) standard. This package was developed primarily using Nihon Kohden (NK) recording exports to EDF (which break the EDF standard in certain ways; for instance, NK EDF exports can include partial final recording records with fewer sample than other records). We expect other clinical systems would introduce different quirks into EDF exports that may necessitate updating this package.

## De-identification operations:
- Remove patient names in the EDF header and optionally in EDF annotations (we find that technicians sometimes place patient names in the annotations, which can contain arbitrary text and should not be assumed to be free of identifying information). Header names are replaced with a user-specified experimental subject code.
- Remove patient birth date in the EDF header
- Drop annotations containing other identifying information, including gendered pronouns and arbitrary regex patterns
- Set recording start times to 1985-01-01 (with relative offsets from the time of the first recording if multiple EDF files from the same subject are processed together to preserve relative timing information)

## Installation:
```
git clone git@github.com:Riley16/clean_eeg.git
cd clean_eeg
# create new conda environment
conda create -n clean_eeg python=3.11
conda activate clean_eeg
pip install .
```

To run unit tests, instead install in editable mode:
```
pip install -e .
pytest
```

To clean a directory of EEG files for one subject:
```
cd clean_eeg
conda activate clean_eeg
python src/clean_eeg/clean_subject_eeg.py --input_path PATH/TO/ALL/SUBJECT/EEG/FILES --output_path OPTIONAL/PATH/TO/OUTPUT/DEIDENTIFIED/FILES --subject_code R1XXXY --first-name John --middle-name Paul --last-name Smith
```
The path to the de-identified files will be printed once the process finishes. Multiple middle names can be specified by separating them with an underscore.

## Dependencies:
- [pyedflib](https://github.com/holgern/pyedflib)
    - our primary EDF IO manager
- [lunapi](https://zzz.bwh.harvard.edu/luna/lunapi/)
    - used to split discontinuous EDF+D files into separate continuous EDF+C files since pyedflib does not support the EDF+D formats
- [edfio](https://github.com/the-siesta-group/edfio)
- [MNE](https://mne.tools/stable/index.html)


## Accessing External or Network Drives (Windows, WSL, macOS)

If your EDF files are stored on an external hard drive, USB device, or network share, you may need to mount the drive so the de-identification tool can access it.
Below are simple instructions for each operating system.

- Windows (PowerShell)

Most external drives appear automatically as a drive letter (e.g., `E:\`, `F:\`, etc.).

List available drives:

`Get-PSDrive -PSProvider FileSystem`

These removable drives can be directly accessed in PowerShell:

`cd E:\path\to\edf_files`

- Windows Subsystem for Linux (WSL / WSL2):

WSL exposes all Windows drives under /mnt. Access an external drive (e.g., `E:\` would be typically mapped automatically to `e` on WSL):
`ls /mnt/e`

Manually mount a drive (if WSL doesn't auto-detect it):
```
# create a mount point if needed:
sudo mkdir -p /mnt/mydrive
# mount:
sudo mount -t drvfs E: /mnt/mydrive
```

Your Python code can now read from /mnt/mydrive.


- macOS (Terminal):

macOS automatically mounts external drives under /Volumes.

List mounted volumes

`ls /Volumes`

Navigate to your drive

`cd /Volumes/MyExternalDrive/path/to/edf_files`

Manually mount a disk (rare cases)

Find the disk:

`diskutil list`

Mount it:

`sudo diskutil mount /dev/disk2s1`

Unmount when done:

`diskutil unmount /Volumes/MyExternalDrive`
