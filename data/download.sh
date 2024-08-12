#!/bin/bash

# This script downloads the data from the source
echo 'Downloading MIND from the source'

mkdir -p MIND/
cd MIND

# file names to grab
declare -a arr=('relations.dict' 'entities.dict' 'test.txt' 'valid.txt' 'train.txt')

for i in "${arr[@]}"; do
    echo ... downloading "$i"
    wget -N -c "https://zenodo.org/records/8117748/files/"${i}"?download=1" -O "${i}"
done

cd ..

echo 'Complete.'
