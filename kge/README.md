# Run model command
## Notebooks
See the following notebooks for KGE [hyperparameter optimization](../Notebooks/1_Sample_OptunaHPO_TransE.ipynb), and [execution](../Notebooks/2_Train-Test-Valid_KGE_Models_MIND.ipynb).

## Command line
   ```
   # run in shell
   bash run.sh train <model_name> <dataset_name> <gpu_num> <folder_out_name> <batch size> <neg_sample_size> <dimensions> <gamma> <alpha> <learningrate> <test_batch_size> <double_entities_emb> <double_relation_emb> <regularization>

   # or in python
   # for more parameters please see codes/run.py Lines 23 - 72
   python run.py --{do_train, do_valid, do_test, do_predict} --data_path <where/data/is> --model {TransE, DistMult, ComplEx, RotatE}ss
   ```