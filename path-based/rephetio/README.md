# MechRepoNet

This repository contains the scripts and files required to build the MechRepoNet Network as
well as build the associated Repositioning Model and perform analysis.

MechRepoNet is fully described in <!--[this preprint publication](https://www.biorxiv.org/content/10.1101/2021.04.15.440028v1.abstract)-->[this preproof publication](https://www.dropbox.com/s/hbby8vfbpdoouex/Bioinfo.-2022-Mayers-Tu.pdf?dl=0).

## Notebooks
See the following notebook for [Rephetio preprocessing and execution](../../Notebooks/7_Get_Predictions_and_Aggregate_results_(Paper_T1).ipynb)

## Organization

This repository is organized as follows.

```
/0_data           # Contains data needed to for use within scripts
    manual        # Data built manually. Most will be included, unless built from proprietary source
    external      # Data acquired from external sources. Not included, but scripts will provide most
/1_code           # Contains all code for running the pipeline. Scripts and notebooks are numbered in order they should be run.
    tools         # contains tools for building
/tools            # simlik to /1_code/tools for compatibility with legacy code

```

## Setting up the environment

A requirements.txt file is provided for running this code. To set up the environment run

`$ pip install -r requirements.txt`

