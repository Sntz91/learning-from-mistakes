# Learning from Mistakes: Visual Failure Monitoring for ICH Detection

This repository contains the paper's () splits, saved test predictions, and scripts to reproduce its results. The diagnostic, monitor, and second-opinion checkpoint weights are available as GitHub releases.

## Evaluate the paper runs

```bash
pip install -r requirements.txt
python evaluate.py
```

This uses the saved predictions in `runs/paper/`; no images or GPU are needed. The numerical results are written to `results/paper/`, including `results.csv`.

## Recreate the data and retrain

Obtain the RSNA Intracranial Hemorrhage Detection () and SinoCT datasets (). Set `RAW_RSNA` and `RAW_SINOCT` at the top of their preparation scripts, then run those commands in order:

```bash
python data/prepare_rsna.py
python data/prepare_sinoct.py
python data/match_rsna_sinoct.py
python data/make_splits.py
```

`python data/prepare_sinoct.py`, `python data/match_rsna_sinoct.py`,, `python data/make_splits.py` are optional, as `splits/rsna/` are already provided in this repro. However, 
running python `data/prepare_rsna.py` remains mandatory for using the dataset class used for 
training.

### Training and Evaluating the Model
On a GPU machine run `python train.py`. This trains the diagnostic model, second opinion, and monitor in order and saves their predictions. To evaluate the new run, point `RUN` and `OUTPUT` in `evaluate.py` to its run and result directories, then run `python evaluate.py`.
