import argparse
import os
from typing import List, Tuple, Optional

import lunapi as lp
from edfio import Edf, EdfAnnotation


def convert_edf_to_continuous_segments(input_file: str, output_dir: str, verbosity: int = 1) -> None:
    inst, segments = luna_open_and_segments(input_file)

    if verbosity > 0:
        print(f"Detected {len(segments)} segment(s).")

    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    for i, (start, stop) in enumerate(segments, start=1):
        # compute gap to previous segment
        if i > 1:
            gap = start - segments[i-2][1]
            if verbosity > 0:
                print(f"  Gap before segment {i}: {gap:.2f} sec")
        tag = base_filename[:]
        if len(segments) > 1:
            tag += f"__seg{i:02d}"
        
        edf_file_no_extension = os.path.join(output_dir, f"{tag}")

        # 1) Mask and write EDF+C for this segment
        print('start:', start, 'stop:', stop)
        luna_write_segment(inst, start, stop, edf_file_no_extension)

        # 2) Still masked — pull annotations directly from Luna
        ann_df = luna_fetch_segment_annots(inst)

        print(ann_df)

        # 3) Clear mask for next iteration
        luna_clear_mask(inst)
        if verbosity > 0:
            print(f"[{i}/{len(segments)}] wrote {edf_file_no_extension}.edf ; "\
                f"{len(ann_df) if ann_df is not None else 0} annotations in-memory")
    if verbosity > 0:
        print(f"Done. Outputs in: {output_dir}")


# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split EDF+D into EDF+C segments with Luna, "
                    "grab segment annotations in-memory, "
                    "then embed them into each segment EDF as EDF+ (using edfio)."
    )
    p.add_argument("-i", "--input", required=True, help="Path to EDF/EDF+ (likely EDF+D) input file")
    p.add_argument("-o", "--outdir", required=True, help="Output directory for segment EDFs")
    # p.add_argument("-p", "--prefix", default="", help="Filename prefix for outputs (e.g., 'subj01_')")
    p.add_argument("--ext", default=".edf", help="Output EDF filename extension (default: .edf)")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not embed annotations; only write EDF+C segments from Luna")
    return p.parse_args()


# -------------------- Luna helpers --------------------
def luna_open_and_segments(edf_path: str) -> Tuple[lp.inst, List[Tuple[float, float]]]:
    """
    Attach the EDF file in Luna and return (instance, [(start, stop), ...]) for contiguous segments.
    """
    proj = lp.proj()
    inst = proj.inst("rec1")
    inst.attach_edf(edf_path)

    # TODO undo luna replacement of spaces in annotation labels with underscores

    inst.proc("SEGMENTS")  # populate segment table
    segs = inst.table("SEGMENTS", "SEG")

    if segs is None or segs.empty:
        # print annotations for debugging
        inst.proc("ANNOTS")

        print("No SEGMENTS detected; available annotations:")
        annotations = inst.edf.fetch_full_annots(['edf_annot'])
        for ann in annotations:
            print(ann)

        # prompt user to check if annotations indicate no discontinuities and continue processing if so
        message = 'Press "Y/y" to continue if the annotations indicate the '\
                  'file is already continuous to proceed with processing...'
        if input(message) in ("Y", "y"):
            # use start/stop of entire recording
            start = float(inst.get("START"))
            stop = float(inst.get("STOP"))
            return inst, [(start, stop)]
        else:
            raise RuntimeError("No segments detected (file may already be continuous or lacks TAL timing).")

    segments = [(float(row["START"]), float(row["STOP"])) for _, row in segs.iterrows()]
    return inst, segments


def luna_write_segment(inst: lp.inst, start: float, stop: float, edf_out: str) -> None:
    """
    Mask to [start, stop) and write a standard continuous EDF (EDF/C).
    """
    inst.proc(f"MASK sec={start}-{stop}")
    inst.proc(f"WRITE edf={edf_out}")
    # keep the mask active so we can query ANNOTS for this segment


def luna_fetch_segment_annots(inst: lp.inst) -> "pandas.DataFrame":
    """
    Assumes a MASK is set. Returns a pandas DataFrame of annotations for the masked segment.
    """
    inst.proc("ANNOTS")       # populate ANNOTS table for current (masked) data
    # df = inst.table("ANNOTS")
    df = inst.edf.fetch_full_annots(['edf_annot'])
    if df is None:
        # Return empty dataframe-like (list) to keep logic simple
        import pandas as pd
        return pd.DataFrame(columns=["CLASS", "INSTANCE", "START", "STOP"])
    return df


def luna_clear_mask(inst: lp.inst) -> None:
    inst.proc("MASK clear")


# -------------------- Annotation conversion --------------------
def _col(df, *names: str) -> Optional[str]:
    """Return first matching column name in df among provided candidates (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        c = lower.get(n.lower())
        if c is not None:
            return c
    return None


def df_to_edfio_annotations(df, seg_start: float, seg_stop: float) -> List[EdfAnnotation]:
    """
    Convert a Luna ANNOTS DataFrame (for a masked segment) into a list of EdfAnnotation,
    rebasing times so the segment starts at t=0.
    Expected columns (names vary a bit across Luna versions):
      - class label:    CLASS or ANNOT
      - instance:       INSTANCE or INST (optional)
      - start (sec):    START
      - stop (sec):     STOP (optional for instantaneous events)
    """
    if df is None or df.empty:
        return []

    c_label = _col(df, "CLASS", "ANNOT")
    c_inst  = _col(df, "INSTANCE", "INST")
    c_start = _col(df, "START")
    c_stop  = _col(df, "STOP")

    if c_label is None or c_start is None:
        # Minimal requirement: label + start
        return []

    anns: List[EdfAnnotation] = []
    seg_len = float(seg_stop - seg_start)

    for _, row in df.iterrows():
        label = str(row[c_label])
        start_abs = float(row[c_start])
        stop_abs = float(row[c_stop]) if (c_stop and not _is_nan(row[c_stop])) else None

        # rebase to segment-relative time
        start_rel = start_abs - seg_start
        stop_rel = (stop_abs - seg_start) if stop_abs is not None else None

        # clip to [0, seg_len]
        if (stop_rel is not None) and (stop_rel <= 0):
            continue  # entirely before
        if start_rel >= seg_len:
            continue  # entirely after
        if start_rel < 0:
            # clip head to 0
            if stop_rel is None:
                start_rel = 0.0
            else:
                # shift start to 0; keep end within segment
                start_rel = 0.0
        if stop_rel is not None and stop_rel > seg_len:
            stop_rel = seg_len

        duration = (stop_rel - start_rel) if stop_rel is not None else None

        # Optionally include instance in text; else just label
        if c_inst and not _is_nan(row[c_inst]):
            text = f"{label}:{row[c_inst]}"
        else:
            text = label

        anns.append(EdfAnnotation(onset=float(start_rel),
                                  duration=(None if duration is None else float(duration)),
                                  text=text))
    return anns


def _is_nan(x) -> bool:
    try:
        from math import isnan
        return isinstance(x, float) and isnan(x)
    except Exception:
        return False


def embed_annots_with_edfio(edf_path: str, annotations: List[EdfAnnotation]) -> None:
    """
    Overwrite the EDF segment with embedded annotations (TALs) as EDF+ continuous.
    """
    from edfio import read_edf
    edf = read_edf(edf_path)
    # Rebuild an Edf object with the same content but new annotations
    print(edf.__dict__.keys())
    edf = Edf(
        edf.signals,
        patient=edf.patient,
        recording=edf.recording,
        start_datetime=edf.start_datetime,
        data_record_duration=edf.data_record_duration,
        reserved=edf.reserved,
        annotations=annotations,
    )
    edf.write(edf_path)  # in-place overwrite


def load_edf_annotations_mne(edf_path, preload=False, verbose=False):
    """
    Load technician/user annotations from an EDF/EDF+ file using MNE.

    Parameters
    ----------
    edf_path : str
        Path to the EDF/EDF+ file.
    preload : bool
        Passed to mne.io.read_raw_edf (you usually don't need to preload to read annotations).
    verbose : bool
        If True, let MNE print info.

    Returns
    -------
    df : pandas.DataFrame
        Columns: ['onset_sec', 'duration_sec', 'description', 'onset_time'].
        - onset_sec/duration_sec are relative to file start (seconds).
        - onset_time is an absolute datetime if available in the file; otherwise None.
        Empty DataFrame if no annotations.
    raw : mne.io.BaseRaw
        The loaded Raw object (so you can keep working with it).
    """
    import mne
    import pandas as pd
    from datetime import timedelta

    # Read EDF; stim_channel=None avoids trying to guess a trigger channel as events
    raw = mne.io.read_raw_edf(
        edf_path,
        preload=preload,
        stim_channel=None,
        verbose=verbose
    )

    ann = raw.annotations  # mne.Annotations or empty
    if ann is None or len(ann) == 0:
        return pd.DataFrame(columns=["onset_sec", "duration_sec", "description", "onset_time"]), raw

    # Onsets are in seconds relative to the file start (MNE sets this appropriately for EDF+)
    df = pd.DataFrame({
        "onset_sec": ann.onset,
        "duration_sec": ann.duration,
        "description": ann.description,
    })

    # If the EDF had an absolute clock (orig_time), add absolute datetimes
    if ann.orig_time is not None:
        base = ann.orig_time  # datetime
        df["onset_time"] = [base + timedelta(seconds=float(t)) for t in df["onset_sec"]]
    else:
        df["onset_time"] = None

    return df, raw


def overwrite_edfD_to_edfC(input_file: str, require_continuous_data: bool = True) -> None:
    """
    Overwrite the reserved field of an EDF+D file to make it EDF+C.
    This does not change the data, just the header field.
    """
    from clean_eeg.load_eeg import RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET, is_edf_continuous, validate_edf_file_path

    if require_continuous_data and not is_edf_continuous(input_file):
        raise ValueError("Input EDF+D file contains discontinuous data; not overwriting to EDF+C. "
                         "Override by setting require_continuous_data=False.")

    validate_edf_file_path(input_file)
    with open(input_file, 'r+b') as f:
        f.seek(RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET)
        f.write(b'EDF+C   ')


def main():
    import os
    args = parse_args()
    input_file = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.outdir)
    os.makedirs(output_dir, exist_ok=True)

    input_dir = os.path.dirname(input_file)
    assert input_dir != output_dir, "Input and output recording directories must be different."

    convert_edf_to_continuous_segments(input_file, output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
