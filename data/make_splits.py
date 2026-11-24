from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

INDEX = Path('data/rsna-processed/index.parquet')
MAPPING = Path('data/sinoct_rsna_mapping.csv')
OUTPUT = Path('splits/rsna')
SEED = 42

def split_patients(frame):
    labels = sorted(c for c in frame if c.startswith('label_'))
    patients = frame.groupby('patient_id')[labels].max().reset_index()

    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, heldout_idx = next(first.split(patients, patients[labels].to_numpy(int)))
    heldout = patients.iloc[heldout_idx]
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(second.split(heldout, heldout[labels].to_numpy(int)))

    return {
        'train': set(patients.iloc[train_idx].patient_id),
        'val': set(heldout.iloc[val_idx].patient_id),
        'test': set(heldout.iloc[test_idx].patient_id),
    }

def write_ids(frame, name):
    path = OUTPUT / f'{name}.parquet'
    frame[['sop_uid']].to_parquet(path, index=False)
    print(f'{path}: {len(frame)} slices')

def main():
    if OUTPUT.exists():
        raise FileExistsError(f'{OUTPUT} already exists.')

    index = pd.read_parquet(INDEX)
    mapping = pd.read_csv(MAPPING)
    matched_ids = set(mapping.rsna_sop_uid)
    if not matched_ids or not matched_ids <= set(index.sop_uid):
        raise ValueError('The mapping must contain RSNA slice IDs from the prepared index.')
    matched_patients = set(index.loc[index.sop_uid.isin(matched_ids), 'patient_id'])
    matched = index[index.patient_id.isin(matched_patients)]
    unmatched = index[~index.patient_id.isin(matched_patients)]

    OUTPUT.mkdir(parents=True)
    write_ids(index, 'all')

    for prefix, frame in (('match-sinoct', matched), ('no-match-sinoct', unmatched)):
        write_ids(frame, f'{prefix}-all')
        groups = split_patients(frame)
        if prefix == 'match-sinoct':
            # Fold matched-domain test patients into training, preserving validation.
            groups['train'] |= groups.pop('test')
        for name, patients in groups.items():
            write_ids(frame[frame.patient_id.isin(patients)], f'{prefix}-{name}')


if __name__ == '__main__':
    main()
