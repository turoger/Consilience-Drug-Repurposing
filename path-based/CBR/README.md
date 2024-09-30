# CBR

Code for the AKBC'20 paper -- [A Simple Approach to Case-Based Reasoning in Knowledge Bases](https://openreview.net/forum?id=AEY9tRqlU7)

## Notebooks
See the following notebook for [CBR preprocessing and execution](../../Notebooks/4_Generate_CBR_Predictions.ipynb)

## Notes
* To run for a new dataset, first a random subgraph around each entity needs to be collected and written to disk.
* Collected 1000 random paths aorund each entity by doing DFS.
* Execution can be found in the notebook above

```
python code/data/get_paths.py --dataset_name <insert_dataset_name> --num_paths_to_collect 1000 --data_dir cbr_akbc_data
```

## Citation
````
@inproceedings{cbr_akbc,
  title = {A Simple Approach to Case-Based Reasoning in Knowledge Bases},
  author = {Das, Rajarshi and Godbole, Ameya and Dhuliawala, Shehzaad and Zaheer, Manzil and McCallum, Andrew},
  booktitle = {AKBC},
  year = 2020
}
````
 
