import io
import logging
import os
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# RSNA uses IDs such as ID_e8005fa8a9 where DICOM expects a numeric UID.
logging.getLogger('pydicom').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message="Invalid value for VR UI: 'ID_.*'", category=UserWarning, module='pydicom')

RAW_RSNA = Path('/run/media/tobias/backup/data/rsna')
OUTPUT = Path('data/rsna-processed')
SHARD_SIZE = 1000
SUBTYPES = ('epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural')

def to_hu(ds):
    raw = ds.pixel_array.astype(np.float32)
    hu = raw * float(getattr(ds, 'RescaleSlope', 1.0)) + float(getattr(ds, 'RescaleIntercept', 0.0))
    return hu.astype(np.int16)

def read_labels():
    """ One row per slice; one column per subtype. """
    labels = pd.read_csv(RAW_RSNA / 'stage_2_train.csv')
    # RSNA one row per slice & subtype: IDs like "ID_abc123_epidural"
    labels[['sop_uid', 'subtype']] = labels.ID.str.rsplit('_', n=1, expand=True)
    # Some data is duplicated
    labels = labels[labels.subtype.isin(SUBTYPES)].drop_duplicates(['sop_uid', 'subtype'])
    labels = labels.pivot(index='sop_uid', columns='subtype', values='Label').fillna(0)
    return labels.rename(columns={name: f'label_{name}' for name in SUBTYPES})

def read_metadata(labels):
    """ one row per labeled slice: path, UIDs, and z position """
    paths = [p for p in (RAW_RSNA / 'stage_2_train').iterdir() if p.suffix == '.dcm']
    if not paths:
        raise FileNotFoundError(RAW_RSNA / 'stage_2_train')
    rows = []
    for path in tqdm(paths, desc='Reading RSNA metadata'):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        uid = str(ds.SOPInstanceUID)
        if uid not in labels.index:
            print(f'Could not find uid: {uid} in labels, skipping.')
            continue
        position = getattr(ds, 'ImagePositionPatient', None)
        z = float(position[2]) if position is not None and len(position) >= 3 else float(getattr(ds, 'SliceLocation', 0))
        rows.append(dict(
            img_path=str(path), 
            sop_uid=uid, 
            patient_id=str(ds.PatientID), 
            series_id=str(ds.SeriesInstanceUID), 
            z_coord=z)
        )
    print(f'{len(paths) - len(rows)} of {len(paths)} DICOMs had no label and were skipped.')
    return pd.DataFrame(rows)

def write_shards(index):
    """ Write readable slices as .npy into tar shards. """
    kept = []
    locations = []
    archive = None
    for row in tqdm(index.itertuples(), total=len(index), desc='Writing HU slices'):
        try:
            image = to_hu(pydicom.dcmread(row.img_path))
        except Exception as exc:
            print(f'Skipping {row.img_path}: {exc}')
            continue

        if len(kept) % SHARD_SIZE == 0:
            if archive is not None:
                archive.close()
            tar_name = f'shard-{len(kept) // SHARD_SIZE:06d}.tar'
            archive = tarfile.open(OUTPUT / tar_name, 'w')
        buf = io.BytesIO()
        np.save(buf, image)
        member = tarfile.TarInfo(f'{row.sop_uid}.npy')
        member.size = buf.tell()
        buf.seek(0)

        start = archive.fileobj.tell()
        archive.addfile(member, buf)
        data_offset = start + 512
        assert archive.fileobj.tell() == data_offset + member.size + (-member.size % 512)
        kept.append(row.Index)
        locations.append((tar_name, data_offset, member.size))
    if archive is not None:
        archive.close()
    index = index.loc[kept].reset_index(drop=True)
    return index.join(pd.DataFrame(locations, columns=['tar', 'data_offset', 'size']))

def verify(index, n=20):
    """ Read some slices back like CTDataset and compare them to DICOMs. """
    for row in index.sample(min(n, len(index)), random_state=0).itertuples():
        fd = os.open(OUTPUT / row.tar, os.O_RDONLY)
        stored = np.load(io.BytesIO(os.pread(fd, row.size, row.data_offset)))
        os.close(fd)
        assert np.array_equal(stored, to_hu(pydicom.dcmread(row.img_path))), f'bad offset for {row.sop_uid}'

def main():
    if list(OUTPUT.glob('shard-*.tar')):
        raise FileExistsError(f'{OUTPUT} already contains shards.')
    OUTPUT.mkdir(parents=True, exist_ok=True)

    labels = read_labels()
    index = read_metadata(labels).merge(labels.reset_index(), on='sop_uid', validate='one_to_one')
    index = index.sort_values(['series_id', 'z_coord']).reset_index(drop=True)
    index['slice_pos'] = index.groupby('series_id').cumcount()

    index = write_shards(index)
    verify(index)
    index.drop(columns='img_path').to_parquet(OUTPUT / 'index.parquet', index=False)
    print(f'Prepared {len(index)} slices at {OUTPUT}')


if __name__=='__main__':
    main()
