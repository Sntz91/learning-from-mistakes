import io
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from prepare_rsna import to_hu

RAW_SINOCT = Path('/run/media/tobias/backup/data/sinoct/ctsinogram/head_ct_dataset_anon')
OUTPUT = Path('data/sinoct-processed')
SHARD_SIZE = 1000

def read_labels():
    labels = pd.read_csv(RAW_SINOCT / 'labels.csv').rename(columns={'patient_id': 'patient_id_'})
    flags = labels.label.str.split(',', expand=True).astype(int)
    labels['label_normal'] = flags[0]
    labels['label_abnormal'] = flags[1]
    return labels.drop(columns='label')

def read_metadata():
    rows = []
    for path in tqdm(sorted(RAW_SINOCT.rglob('*.dcm')), desc='Reading SinoCT metadata'):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        slice_pos = int(''.join(filter(str.isdigit, path.stem)) or '0')
        rows.append(dict(
            key=str(ds.SOPInstanceUID).replace('.', '_'),
            img_path=str(path),
            patient_id=str(ds.PatientID),
            patient_id_=path.parents[1].name,
            sop_uid=str(ds.SOPInstanceUID),
            series_id=str(ds.SeriesInstanceUID),
            slice_pos=slice_pos,
            z_coord=float(getattr(ds, 'SliceLocation', slice_pos)),
        ))
    return pd.DataFrame(rows)

def write_shards(index):
    kept = []
    locations = []
    archive = None
    for row in tqdm(index.itertuples(), total=len(index), desc='Writing SinoCT HU slices'):
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
        locations.append((tar_name, start, data_offset, member.size))

    if archive is not None:
        archive.close()
    index = index.loc[kept].reset_index(drop=True)
    return index.join(pd.DataFrame(locations, columns=['tar', 'offset', 'data_offset', 'size']))

def verify(index, n=20):
    for row in index.sample(min(n, len(index)), random_state=0).itertuples():
        fd = os.open(OUTPUT / row.tar, os.O_RDONLY)
        stored = np.load(io.BytesIO(os.pread(fd, row.size, row.data_offset)))
        os.close(fd)
        assert np.array_equal(stored, to_hu(pydicom.dcmread(row.img_path))), f'bad offset for {row.sop_uid}'

def main():
    if list(OUTPUT.glob('shard-*.tar')):
        raise FileExistsError(f'{OUTPUT} already contains shards.')
    OUTPUT.mkdir(parents=True, exist_ok=True)

    index = read_metadata().merge(read_labels(), on='patient_id_', how='inner')
    index = index.sort_values(['series_id', 'slice_pos']).reset_index(drop=True)
    index = write_shards(index)
    verify(index)
    index.drop(columns='img_path').to_parquet(OUTPUT / 'index.parquet', index=False)
    print(f'Prepared {len(index)} SinoCT slices at {OUTPUT}')


if __name__ == '__main__':
    main()
