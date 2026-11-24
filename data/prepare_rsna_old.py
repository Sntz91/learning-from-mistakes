"""Convert raw RSNA ICH DICOMs into the processed archive used for training."""

from pathlib import Path
import io
import tarfile

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

RAW_RSNA = Path("/path/to/rsna")
OUTPUT = Path("data/rsna-processed")
SHARD_SIZE = 1000

SUBTYPES = ("epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural")


def main():
    labels = pd.read_csv(RAW_RSNA / "stage_2_train.csv")
    # RSNA labels are one row per image and subtype.
    labels[["sop_uid", "subtype"]] = labels.ID.str.rsplit("_", n=1, expand=True)
    labels = labels[labels.subtype.isin(SUBTYPES)].drop_duplicates(["sop_uid", "subtype"])
    labels = labels.pivot(index="sop_uid", columns="subtype", values="Label").fillna(0)
    labels = labels.rename(columns={name: f"label_{name}" for name in SUBTYPES})
    labels.index = labels.index.astype(str)

    image_paths = sorted((RAW_RSNA / "stage_2_train").glob("*.dcm"))
    if not image_paths:
        raise FileNotFoundError(RAW_RSNA / "stage_2_train")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in tqdm(image_paths, desc="Reading RSNA metadata"):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        uid = str(ds.SOPInstanceUID)
        if uid not in labels.index:
            continue
        position = getattr(ds, "ImagePositionPatient", None)
        z = float(position[2]) if position is not None and len(position) >= 3 else float(getattr(ds, "SliceLocation", 0))
        rows.append(dict(img_path=str(path), sop_uid=uid, patient_id=str(ds.PatientID),
                         series_id=str(ds.SeriesInstanceUID), z_coord=z))
    index = pd.DataFrame(rows).merge(labels.reset_index(), on="sop_uid", validate="one_to_one")
    index = index.sort_values(["series_id", "z_coord"]).reset_index(drop=True)
    index["slice_pos"] = index.groupby("series_id").cumcount()

    shards = []
    archive = None
    for i, row in tqdm(index.iterrows(), total=len(index), desc="Writing HU slices"):
        if i % SHARD_SIZE == 0:
            if archive is not None:
                archive.close()
            tar_name = f"shard-{i // SHARD_SIZE:06d}.tar"
            archive = tarfile.open(OUTPUT / tar_name, "w")
        ds = pydicom.dcmread(row.img_path)
        hu = (ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0))
              + float(getattr(ds, "RescaleIntercept", 0.0))).astype(np.int16)
        buf = io.BytesIO()
        np.save(buf, hu)
        member = tarfile.TarInfo(f"{row.sop_uid}.npy")
        member.size = buf.tell()
        offset = archive.fileobj.tell()
        buf.seek(0)
        archive.addfile(member, buf)
        end = archive.fileobj.tell()
        padded = member.size + (-member.size % 512)
        shards.append((tar_name, offset, end - padded, member.size))
    if archive is not None:
        archive.close()
    index[["tar", "offset", "data_offset", "size"]] = shards
    index = index.drop(columns="img_path")

    # The paper's patient split is supplied; every split UID must exist in the data.
    available = set(index.sop_uid)
    for split in Path("splits/rsna").glob("*.parquet"):
        missing = set(pd.read_parquet(split).sop_uid) - available
        if missing:
            raise ValueError(f"{split}: {len(missing)} images missing from the raw RSNA data")
    index.to_parquet(OUTPUT / "index.parquet", index=False)
    print(f"Prepared {len(index)} slices at {OUTPUT}")


if __name__ == "__main__":
    main()
