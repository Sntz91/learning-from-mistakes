import hashlib
import io
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


RSNA = Path('data/rsna-processed')
SINOCT = Path('data/sinoct-processed')
OUTPUT = Path('data/sinoct_rsna_mapping.csv')


def slice_hashes(root, index):
    for tar_name, rows in tqdm(index.groupby('tar'), desc=f'Hashing {root.name}'):
        with tarfile.open(root / tar_name) as archive:
            for row in rows.itertuples(index=False):
                member = archive.extractfile(f'{row.sop_uid}.npy')
                image = np.load(io.BytesIO(member.read()), allow_pickle=False)
                yield hashlib.md5(image.tobytes()).digest(), row


def main():
    sinoct = pd.read_parquet(SINOCT / 'index.parquet')
    rsna = pd.read_parquet(RSNA / 'index.parquet')

    # SinoCT is smaller, so keep its hashes in memory and stream through RSNA.
    sinoct_by_hash = defaultdict(list)
    for digest, row in slice_hashes(SINOCT, sinoct):
        sinoct_by_hash[digest].append(row)

    matches = []
    for digest, row in slice_hashes(RSNA, rsna):
        for other in sinoct_by_hash.get(digest, ()):
            matches.append(dict(
                rsna_sop_uid=row.sop_uid,
                rsna_series_id=row.series_id,
                rsna_patient_id=row.patient_id,
                sinoct_sop_uid=other.sop_uid,
                sinoct_series_id=other.series_id,
                sinoct_patient_id=other.patient_id,
            ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matches, columns=(
        'rsna_sop_uid', 'rsna_series_id', 'rsna_patient_id',
        'sinoct_sop_uid', 'sinoct_series_id', 'sinoct_patient_id',
    )).to_csv(OUTPUT, index=False)
    print(f'Found {len(matches)} matching slice pairs; saved {OUTPUT}')


if __name__ == '__main__':
    main()
