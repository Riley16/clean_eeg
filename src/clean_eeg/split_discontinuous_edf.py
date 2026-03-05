import argparse
import os
from typing import List, Tuple

import lunapi as lp


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
        luna_write_segment(inst, start, stop, edf_file_no_extension)

        # 2) Still masked — pull annotations directly from Luna
        ann_df = luna_fetch_segment_annots(inst)

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
                    "then embed them into each segment EDF as EDF+."
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
