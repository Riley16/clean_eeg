import numpy as np
from clean_eeg.load_eeg import load_edf


# compare EDF files for equality
def compare_edf_files(file1, file2,
                      load_method='pyedflib',
                      physical_range_rel_tol=0.0,
                      compare_signals=True,
                      verbosity=0):
    """
    Compare two EDF files for equality.

    Parameters:
        file1 (str): Path to the first EDF file.
        file2 (str): Path to the second EDF file.
        load_method (str): Method to load EDF files ('edfio', 'pyedflib', or 'mne').

    Returns:
        bool: True if files are equal, False otherwise.
    """
    data1 = load_edf(file1, load_method=load_method, preload=compare_signals, read_digital=True)
    data2 = load_edf(file2, load_method=load_method, preload=compare_signals, read_digital=True)

    # compare headers, signal headers, signals, and annotations separately
    if load_method == 'edfio':
        raise NotImplementedError("Comparison for edfio is not implemented yet.")
    elif load_method == 'pyedflib':
        return compare_edf_pyedflib(data1, data2,
                                    physical_range_rel_tol=physical_range_rel_tol,
                                    verbosity=verbosity)
    elif load_method == 'mne':
        # MNE Raw objects do not expose all header info, making direct comparison less informative
        raise NotImplementedError("Comparison for MNE is not implemented yet.")
    else:
        raise ValueError("Invalid load method specified. Use 'pyedflib'.")


def compare_edf_pyedflib(data1, data2, physical_range_rel_tol=1e-03, verbosity=0):
    if not isinstance(data1, dict) or not isinstance(data2, dict):
        raise ValueError("Data loaded with pyedflib should be a dictionary.")
    
    headers_equal = compare_pyedflib_headers(data1['header'], data2['header'], verbosity=verbosity)
    
    signal_headers_equal = compare_pyedflib_signal_headers(data1['signal_headers'],
                                                           data2['signal_headers'],
                                                           physical_range_rel_tol=physical_range_rel_tol,
                                                           verbosity=verbosity)
    
    if np.logical_xor(data1['signals'] is None, data2['signals'] is None):
        raise ValueError("Signals present in one but not both data dictionaries.")
    if data1['signals'] is not None and data2['signals'] is not None:
        signals_equal = compare_pyedflib_signals(data1['signals'], data2['signals'], verbosity=verbosity)
    
    if np.logical_xor('annotations' in data1, 'annotations' in data2):
        raise ValueError("Annotations present in one but not both data dictionaries.")
    if 'annotations' in data1 and 'annotations' in data2:
        annotations_equal = compare_pyedflib_annotations(data1['annotations'],
                                                         data2['annotations'],
                                                         verbosity=verbosity)
    else:
        annotations_equal = True
    return headers_equal and signal_headers_equal and signals_equal and annotations_equal


def compare_pyedflib_headers(header1, header2, verbosity=0):
    is_equal = True
    if not isinstance(header1, dict) or not isinstance(header2, dict):
        raise ValueError("Header data should be a dictionary.")
    if header1 != header2:
        if verbosity > 0:
            print(f"Headers differ:\n{header1}\n{header2}")
        is_equal = False
    return is_equal


def compare_pyedflib_signal_headers(signal_headers1, signal_headers2,
                                    physical_range_rel_tol=1e-03,
                                    verbosity=0):
    is_equal = True
    if not isinstance(signal_headers1, list) or not isinstance(signal_headers2, list):
        raise ValueError("Signal headers should be a list.")
    if len(signal_headers1) != len(signal_headers2):
        if verbosity > 0:
            print(f"Signal headers length differ: {len(signal_headers1)} vs {len(signal_headers2)}")
        is_equal = False
    else:
        def isclose_key_value(di1, di2, key, rtol):
            return key in di1 and key in di2 and np.isclose(di1[key], di2[key], rtol=rtol)
        for sh1, sh2 in zip(signal_headers1, signal_headers2):
            physical_range_keys = ['physical_min', 'physical_max']
            if physical_range_rel_tol > 0:
                # separately check physical ranges for approximate equality
                # luna can slightly modify physical ranges
                for sh1, sh2 in zip(signal_headers1, signal_headers2):
                    for key in physical_range_keys:
                        if not isclose_key_value(sh1, sh2, key, rtol=physical_range_rel_tol):
                            print(f"{key} differ:", sh1, "vs", sh2)
                            is_equal = False
                sh1 = {k: v for k, v in sh1.items() if k not in physical_range_keys}
                sh2 = {k: v for k, v in sh2.items() if k not in physical_range_keys}
            if sh1 != sh2:
                if verbosity > 0:
                    print(f"Signal headers differ:\n{sh1}\n{sh2}")
                is_equal = False
    return is_equal


def compare_pyedflib_signals(signals1, signals2,
                             match_initial_values_only=True,
                             verbosity=0):
    is_equal = True
    signals1 = np.array(signals1)
    signals2 = np.array(signals2)
    if not isinstance(signals1, np.ndarray) or not isinstance(signals2, np.ndarray):
        print(signals1)
        raise ValueError("Signals should be numpy arrays.")
    if signals1.shape != signals2.shape:
        if verbosity > 0:
            print(f"Signals shape differ: {signals1.shape} vs {signals2.shape}")
        if match_initial_values_only:
            if len(signals1) != len(signals2):
                if verbosity > 0:
                    print(f"Number of channels differ: {len(signals1)} vs {len(signals2)}")
                return False
            min_length = min(signals1.shape[1], signals2.shape[1])
            max_length = max(signals1.shape[1], signals2.shape[1])
            signals1 = signals1[:, :min_length]
            signals2 = signals2[:, :min_length]
            if verbosity > 0:
                print(f"WARNING: Comparing only first {min_length} samples of each signal (dropping {max_length - min_length} samples from comparison).")
        else:
            return False
    
    if not np.array_equal(signals1, signals2):
        if verbosity > 0:
            print("Signals differ.")
            print('Proportion of values different:', np.mean(signals1 != signals2))
            print('Proportion not close:', np.mean(~np.isclose(signals1, signals2, rtol=1e-03)))
            print(signals1.shape)
            print('Proportion of channels different:', np.mean(np.any(signals1 != signals2, axis=1)))
        is_equal = False
    return is_equal


def compare_pyedflib_annotations(annotations1, annotations2, verbosity=0):
    is_equal = True
    if not isinstance(annotations1, tuple) or not isinstance(annotations2, tuple):
        raise ValueError(f"Annotations should be a tuple of arrays. annotations1: {type(annotations1)}, annotations2: {type(annotations2)}")
    for arr1, arr2 in zip(annotations1, annotations2):
        if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
            raise ValueError("Each annotation item should be a numpy array.")
    annotations1 = np.array(annotations1).T
    annotations2 = np.array(annotations2).T
    if len(annotations1) != len(annotations2):
        if verbosity > 0:
            print(f"Annotations length differ: {len(annotations1)} vs {len(annotations2)}")
        is_equal = False
    else:
        for ann1, ann2 in zip(annotations1, annotations2):
            if np.any(ann1 != ann2):
                if verbosity > 0:
                    print(f"Annotations differ:\n{ann1}\n{ann2}")
                is_equal = False
    return is_equal


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load an EDF file using pyedflib or MNE.")
    parser.add_argument("--path1", type=str, required=True, help="Path to first EDF file to compare")
    parser.add_argument("--path2", type=str, required=True, help="Path to second EDF file to compare")
    parser.add_argument("--load-method", type=str, default="pyedflib", help="Method to load EDF files: 'pyedflib'")
    parser.add_argument("--lazy-load", action="store_true", help="Preload EEG into memory")
    parser.add_argument("--raise-errors", action="store_true", help="Raise errors instead of warnings for debugging")
    parser.add_argument("--verbosity", type=int, default=1, help="Enable verbose output")
    args = parser.parse_args()

    compare_edf_files(args.path1, args.path2,
                      load_method=args.load_method,
                      compare_signals=not args.lazy_load,
                      physical_range_rel_tol=1e-3,
                      verbosity=args.verbosity)
