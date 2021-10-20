⟩#!/bin/bash

# assuming wget and zip are available
wget https://zenodo.org/record/4836009/files/data.zip
unzip data.zip

# copy the data-specific README to data folder
wget -O data/README.md  https://zenodo.org/record/4836009/files/README.md
rm data.zip
