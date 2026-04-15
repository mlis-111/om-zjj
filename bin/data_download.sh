#!/bin/bash
BASE_URL="https://zenodo.org/records/13119437/files"
SAVE_DIR="/data/zjj/om-zjj/data/bio-ml-2024"
mkdir -p "$SAVE_DIR"

files=(
  "ncit-doid.zip"
  "omim-ordo.zip"
  "snomed-fma.body.zip"
  "snomed-ncit.neoplas.zip"
  "snomed-ncit.pharm.zip"
)

for f in "${files[@]}"; do
  echo "Downloading $f ..."
  wget -c "${BASE_URL}/${f}?download=1" -O "${SAVE_DIR}/${f}"
  echo "Extracting $f ..."
  unzip -q "${SAVE_DIR}/${f}" -d "${SAVE_DIR}"
  rm "${SAVE_DIR}/${f}"
done

echo "Done. Contents:"
ls "$SAVE_DIR"