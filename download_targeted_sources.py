"""Recreate the reviewed public-source image files from the committed audit CSV."""

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "outputs" / "flower25_targeted_v2" / "source_audit.csv"
ALLOWED_HOSTS = {
    "upload.wikimedia.org", "thumb.wikimedia.org",
    "inaturalist-open-data.s3.amazonaws.com", "static.inaturalist.org",
}


def main():
    rows = pd.read_csv(AUDIT)
    for row in rows.itertuples(index=False):
        if row.status != "accepted":
            continue
        url = urllib.parse.urlparse(row.image_url)
        if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"Unexpected image host: {row.image_url}")
        target = ROOT / row.path
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == row.file_sha256:
            continue
        request = urllib.request.Request(row.image_url, headers={"User-Agent": "LingjingFlowerResNet/0.1 educational image audit"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != row.file_sha256:
            raise ValueError(f"Source changed: {row.page_url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        with Image.open(target) as image:
            image.verify()
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
