# Drug Repositioning using Consilience of Knowledge Graph Completion Methods
This repository is a fork of [Sun et. al](https://openreview.net/forum?id=HkgEQnRqYQ) implementation of four knowledge graph embedding models. Here we apply the aforementioned algorithms to a biomedical knowledge graph called MIND ([**M**echRepoNet](https://github.com/SuLab/MechRepoNet) with [DrugCentral](https://drugcentral.org/) **ind**ications). We report the results of our analysis in this [preprint](https://www.biorxiv.org/content/10.1101/2023.05.12.540594v1).

## Modifications to the original repository
* Added code to output raw embeddings to extract predictions. This can be seen with the `--do_predict` flag in `codes/run.py`.
* Added Notebooks folder that encapsulates analysis done on the MIND dataset
* Added methods, `Notebooks/score_utils.py`,to process and translate raw embeddings into human readable entities and relations

## Usage instructions
1. Please see the original PyTorch implementation instructions
2. Download the [MIND dataset](https://tobehostedsomewhere) to `./data`
3. Install requirements into python virtual environemnt
   ```
   # run in shell
   python3 -m venv <virtual_env_name_here>
   source virtual_env_name_here/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ``` 
5. Train/Test
   ```
   # run in shell
   bash run.sh train <model_name> <dataset_name> <gpu_num> <folder_out_name> <batch size> <neg_sample_size> <dimensions> <gamma> <alpha> <learningrate> <test_batch_size> <double_entities_emb> <double_relation_emb> <regularization>

   # or in python
   # for more parameters please see codes/run.py Lines 23 - 72
   python run.py --{do_train, do_valid, do_test, do_predict} --data_path <where/data/is> --model {TransE, DistMult, ComplEx, RotatE}
   ```

## Citation

If you use the codes, please cite the original [paper](https://openreview.net/forum?id=HkgEQnRqYQ) by Sun et al:
```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```
