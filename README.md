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
pip install -e ".[test]"
pytest
```

## Usage

To de-identify a directory of EEG files for one subject (modifies files in place by default):
```
cd clean_eeg
conda activate clean_eeg
python src/clean_eeg/clean_subject_eeg.py \
  --input_path /path/to/subject/edf/files \
  --subject_code SUBJECT_CODE \
  --first_name FIRST_NAME \
  --middle_name MIDDLE_NAME \
  --last_name LAST_NAME
```

By default, EDF files are de-identified in place — headers are modified directly in the original files and a separate annotation stub is created alongside each file. This avoids rewriting signal data and is significantly faster.

To write de-identified copies to a separate directory instead, use `--copy_path`:
```
python src/clean_eeg/clean_subject_eeg.py \
  --input_path /path/to/subject/edf/files \
  --copy_path /path/to/output/directory \
  --subject_code SUBJECT_CODE \
  --first_name FIRST_NAME \
  --middle_name MIDDLE_NAME \
  --last_name LAST_NAME
```

If `--copy_path` is used without a value, de-identified files are written to a `deidentified_eeg_files` subdirectory within the input path.

If the subject has no middle name, pass an empty string:
```
  --middle_name ""
```

If the subject has multiple middle names, separate them with underscores:
```
  --middle_name MIDDLE1_MIDDLE2
```

Any required arguments not provided on the command line will be prompted for interactively. The path to the de-identified files will be printed once the process finishes.

## Log files

The pipeline writes a log file (`log.out`) to the current working directory. All console output is duplicated to this file, with patient name parts automatically scrubbed (replaced with `[PHI_REDACTED]`). After the pipeline finishes, the log is also copied to the output directory alongside the de-identified EDF files.

If the pipeline encounters an error, it will print the log file path and ask you to send it to the data management team for debugging. Because PHI is scrubbed from the log, it is safe to share.

After a successful run, a final redaction pass (fuzzy matching and nickname variants) is applied to the log file to catch any name fragments that may have been missed during streaming output.

## Dependencies:
- [pyedflib](https://github.com/holgern/pyedflib) — primary EDF I/O and header manipulation
- [lunapi](https://zzz.bwh.harvard.edu/luna/lunapi/) — splitting discontinuous EDF+D files into continuous EDF+C segments (pyedflib does not support EDF+D)
- [numpy](https://numpy.org/) — array operations for signal data
- [presidio-analyzer](https://github.com/microsoft/presidio) / [presidio-anonymizer](https://github.com/microsoft/presidio) — NLP entity detection and redaction
- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) — fuzzy name matching via Levenshtein distance
- [nicknames](https://github.com/carltonnorthern/nickname-and-diminutive-names-lookup) — nickname variant expansion (e.g., John → Johnny)
- [regex](https://github.com/mrabarnett/mrab-regex) — advanced regex support
- [tqdm](https://github.com/tqdm/tqdm) — progress bars


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
