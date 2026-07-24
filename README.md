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

Once installed via `pip install .` (or `pip install -e .`), the pipeline can be invoked with `python -m clean_eeg.clean_subject_eeg` from any working directory — you do not need to `cd` into the `clean_eeg` repo.

To de-identify a directory of EEG files for one subject (modifies files in place by default):
```
conda activate clean_eeg
python -m clean_eeg.clean_subject_eeg \
  --input_path /path/to/subject/edf/files \
  --subject_code SUBJECT_CODE \
  --first_name FIRST_NAME \
  --middle_name MIDDLE_NAME \
  --last_name LAST_NAME
```

By default, EDF files are de-identified in place — headers are modified directly in the original files and a separate annotation stub is created alongside each file. This avoids rewriting signal data and is significantly faster.

To write de-identified copies to a separate directory instead, use `--copy_path`:
```
python -m clean_eeg.clean_subject_eeg \
  --input_path /path/to/subject/edf/files \
  --copy_path /path/to/output/directory \
  --subject_code SUBJECT_CODE \
  --first_name FIRST_NAME \
  --middle_name MIDDLE_NAME \
  --last_name LAST_NAME
```

Alternatively, you can still invoke the script directly from a clone of the repo:
```
cd clean_eeg
python src/clean_eeg/clean_subject_eeg.py --input_path ... --subject_code ... ...
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

At the end of a successful run the pipeline writes a `deidentify.json` sidecar into the output directory (recording the manifest of de-identified files with fast-hash checksums), prints a "Human review needed" block for any annotations that contained PHI-adjacent text or header fields that were truncated on write, and asks whether to transfer the files to the CML rhino server. Answering `y` invokes the transfer step described below; answering `n` (or pressing Enter) exits, and you can run `transfer-subject-eeg <output_dir>` at any time to upload.

Re-invoking `clean_subject_eeg` on a directory that already has a `deidentify.json` short-circuits straight to the "already done, skip to transfer?" prompt. Pass `--force` to re-run de-identification from scratch instead. If 5 EDF files in a row fail to load, the pipeline aborts with a message pointing at the `--force_load_all` escape hatch — usually a systematic input-directory issue (wrong export format, permissions, truncated USB dump) rather than genuine per-file corruption.

## Transferring de-identified files to the CML server

```
transfer-subject-eeg /path/to/deidentified/output/dir
```

Runs a preflight that refuses to upload unless the directory looks fully de-identified: `deidentify.json` present, no non-empty `quarantine/` subdir, every EDF matches the de-identified filename pattern, every header shows the redacted patient fields, the site letter is known, and a spot-check hash matches what was recorded at de-id time. If any check fails, the tool prints the reasons and exits without touching the network.

On pass, prints the composed `rsync` (or `scp` fallback) command and asks for confirmation. Uses `rsync --partial` so an interrupted upload can be safely resumed by simply re-running the command (rsync's delta algorithm block-checksums the shared prefix on resume).

Options:
- `--dry-run` — preflight and print the composed commands without invoking anything
- `--user USER` — SSH username (defaults to `$USER`)
- `--yes`/`-y` — skip the interactive confirmation prompt

## Post-upload audit

```
audit-subject-eeg /data10/RAM/incoming/{SITE}/{SUBJECT}/all_clinical_eeg
```

Runs an independent per-subject PHI audit against the uploaded directory: header-residue scan, annotation-dictionary scan, byte-geometry checks, hash comparison against the pipeline's `deidentify.json` manifest, log-file scan, and more. Produces `edf_audit.json` plus a rendered notebook + HTML report in the subject dir. Data analysts run this on the cluster after upload; it is not intended for hospital-site operators (the name-dictionary scan is deliberately noisy for review purposes).

Pass `--parent /data10/RAM/incoming/{SITE}` to audit every subject subfolder in one pass. See `audit-subject-eeg --help` for the full check inventory and options.

## Inspecting EDF headers (debugging)

The package ships a `print-edf-header` command for dumping the raw bytes and parsed values of every EDF header field. It works even when `pyedflib` refuses to open the file (which is typically when you'd reach for it — e.g. a Nihon Kohden export with empty/blank numeric fields). Operates on a single `.edf` file or every `.edf` in a directory.

```
conda activate clean_eeg
print-edf-header /path/to/file.edf
print-edf-header /path/to/folder_of_edfs/
print-edf-header /path/to/file.edf --signals 0,1,5     # only show these signals
print-edf-header /path/to/file.edf --no-signals        # main header only
```

Equivalent module form (useful from inside Python projects):

```
python -m clean_eeg.print_edf_header /path/to/file.edf
```

For each file, the command prints (i) the main header field-by-field with offsets, raw bytes, and parsed values; (ii) per-signal headers; (iii) derived geometry and a verdict on whether the on-disk filesize matches the header. Empty or unparseable numeric fields are surfaced as `<empty>` / `<unparseable: ...>` rather than crashing the script.

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
