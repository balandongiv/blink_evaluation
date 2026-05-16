# blink_evaluation

Small MNE-based workflow for creating and checking blink annotations from raw EEG data.

## What this repo contains

- `load_sample_file.py` to open a raw recording, annotate it manually, and save the result
- `ground_truth_annotations.csv` with plain `onset`, `duration`, `description` columns
- `ground_truth_annotations.fif` with the saved MNE annotations
- `file/check_ground_truth_annotations.py` to compare one case file at a time and open a blocking review plot
- `case_1.csv`, `case_2.csv`, `case_3.csv` as comparison cases

## Annotation format

The exported CSV uses this format:

```csv
onset,duration,description
18.472012,1.240545,eye_blink
```

Notes:

- `onset` and `duration` are in seconds
- `description` is the label you assign during manual review
- there is no timestamp or date formatting

## Create ground-truth annotations

Run:

```bash
python load_sample_file.py
```

This will:

1. load the sample raw FIF file
2. filter and resample it
3. open the MNE plot window with `block=True`
4. let you add manual annotations
5. save the annotations to `ground_truth_annotations.csv`

## Check one case at a time

Use the checker in the `file/` folder:

```bash
python file/check_ground_truth_annotations.py case_1
python file/check_ground_truth_annotations.py case_2
python file/check_ground_truth_annotations.py case_3
```

If you do not pass an argument, it defaults to `case_2`.

The checker does two things:

1. compares the selected `case_*.csv` against `ground_truth_annotations.fif`
2. opens an interactive MNE plot with the annotations attached using `block=True`

Close the plot window when you are done reviewing.

## Dependencies

The scripts depend on:

- `mne`
- `numpy`

## Project layout

```text
blink_evaluation/
├── file/
│   └── check_ground_truth_annotations.py
├── load_sample_file.py
├── ground_truth_annotations.csv
├── ground_truth_annotations.fif
├── case_1.csv
├── case_2.csv
├── case_3.csv
└── README.md
```
