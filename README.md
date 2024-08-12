# Drug Repurposing using Consilience of Knowledge Graph Completion Methods
This repository is a fork of [Sun et. al](https://openreview.net/forum?id=HkgEQnRqYQ) implementation of four knowledge graph embedding models. Here we apply the aforementioned algorithms to a biomedical knowledge graph called MIND ([**M**echRepoNet](https://github.com/SuLab/MechRepoNet) with [DrugCentral](https://drugcentral.org/) **ind**ications). We report the results of our analysis in this [preprint](https://www.biorxiv.org/content/10.1101/2023.05.12.540594v3).

## Modifications to the original repository
* Added code to output raw embeddings in order to extract predictions. This can be seen with the `--do_predict` flag in `kge/codes/run.py`.
* Added Notebooks folder that encapsulates analysis done on the MIND dataset.
* Added methods, `Notebooks/score_utils.py`, to process and translate raw embeddings into human readable entities and relations.

## Usage instructions
1. Please see the original PyTorch implementation instructions
2. Download the [MIND dataset](https://zenodo.org/records/8117748) to `./data` with
```bash
   # run in shell
   bash download.sh
```
3. Install requirements into python virtual environment
   ```bash
   # run in shell
   mamba create -f environment.yml
   mamba activate kge
   
   ``` 
4. Run notebooks in `./Notebooks`

## Citation

If you use the codes, please cite the following papers:
* [RotatE](https://openreview.net/forum?id=HkgEQnRqYQ) by Sun _et al._
* [ProbCBR](https://arxiv.org/abs/2010.03548) by Das _et al._
* [MechRepoNet](https://doi.org/10.1093/bioinformatics/btac205) by Mayers, Tu _et al._
* [MIND](https://doi.org/10.1101/2023.05.12.540594) by Tu, Sinha _et al._
```
@article{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}

@article{das2020probabilistic,
	title = {Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion},
	author = {Das, Rajarshi and Godbole, Ameya and Monath, Nicholas and Zaheer, Manzil and McCallum, Andrew},
   journal = {arXiv},
	year = {2020},
   url = {http://arxiv.org/abs/2010.03548},
}

@article{mayers2022design,
	title = {Design and application of a knowledge network for automatic prioritization of drug mechanisms},
   author = {Mayers, Michael and Tu, Roger and Steinecke, Dylan and Li, Tong Shu and Queralt-Rosinach, Núria and Su, Andrew I.},
	journal = {Bioinformatics (Oxford, England)},	
   year = {2022},
   doi = {10.1093/bioinformatics/btac205},
}

@article{tu2023drug
   title = {Drug Repurposing using Consilience of Knowledge Graph Completion Methods},
   author = {Tu, Roger and Sinha, Meghamala and Gonzalez, Carolina and Hu, Eric and Dhuliawala, Shehzaad Z. and McCallum, Andrew and Su, Andrew I.},
   journal = {BioRxiv},
   year = {2023},
   doi = {10.1101/2023.05.12.540594}
}
```
