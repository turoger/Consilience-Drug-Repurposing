# Prob-CBR

Code for the EMNLP-Findings paper -- [Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion](https://arxiv.org/abs/2010.03548)

## Notebooks
See the following notebook for [prob-CBR preprocessing and execution](../../Notebooks/5_Generate_pCBR_Predictions.ipynb)

## Notes
* A subgraph is collected around each entity in the KG. In practice, we gather a set of paths around each entity. This needs to be done once offline. 
* As the number of entities in MIND is large, please run the preprocessing script as highlighted in the notebooks to generate the subgraph. 
* Preprocessing *MUST* be done in this order: 
  * `--create_vocab`
  * `--calculate_ent_similarity`
  * `--get_paths_parallel`
  * `--combine_paths`
  * `--calculate_prior_map_parallel`
  * `--combine_prior_map`
  * `--calculate_precision_map_parallel`

* For parallelization, use the `current_job` and `total_jobs` arguments to run parallel process.

## Citation

```
@inproceedings{prob_cbr,
  title = {Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion},
  author = {Das, Rajarshi and Godbole, Ameya and Monath, Nicholas and Zaheer, Manzil and McCallum, Andrew},
  booktitle = {Findings of EMNLP},
  year = 2020
}
```
