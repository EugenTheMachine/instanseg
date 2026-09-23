# KAGGLE CHANGELOG

## GENERAL DESCRIPTION

Within this task you must implement the functional utils to be able to save intermediate model checkpoints to Kaggle using corresponding python package and credentials file. You must implement and test every requirement described further in this file. My Kaggle token is currently stored in the `kaggle_token.txt` file locally.

## REQUIREMENTS

### GENERAL REQUIREMENTS

Here is a general step-by-step workflow that you must follow when completing this task. You can only proceed to the next step after you have successfully completed the current step and verified that it is actually fully accomplished. You must not proceed to the next step if you have not verified that the current step is fully accomplished:

1. Analyze this spec thoroughly and carefully to make sure that you have read everything and did not miss something.
2. Analyze the existing codebase to understand the best way to implement the code changes.
3. Plan your work step-by-step to ensure that you implement every requirement into the current codebase in the most efficient way and without breaking it.
4. Implement every requirement by implementing the corresponding code changes.
5. After you have implemented in code every requirement from this file, create the corresponding tests which cover every requirement, so that you can ensure that everything works properly, as expected, and does not cause errors.
6. Run all the tests to ensure everything works as expected. If you see that some tests do not pass, correct the corresponding code parts and re-run the tests again. Repeat this process until every test passes successfully. If at some point you face the execution error which is not related to the incorrect code you produced, but related to the fact that the code executor cannot be reached, stop any actions immediately and tell me explciity that "tests cannot be executed because of some internal error". Do not proceed until I explicitly tell you to do so and await my instructions.
7. Create a training Jupyter notebook which I could execute in Kaggle myself to see whether everything works as expected. Then tell me what file it is - and I will perform the testing myself. If I come back to you with any errors, you will have to fix them, so that I could then re-run the tests.
8. After you have executed and passed all the tests, and I also passed all the tests myself in Kaggle, verify the scope of the work you did with what is required in this file to ensure you actually implemented everything required. If something is missing, implement and/or test it and then verify again. Keep verifying until you ensure that you actually verified and tested everything required.
9. After successfully passing full verification from the previous step, explicitly tell me that "all the requirements have been properly implemented and tested. The task is completed".

### 1. Kaggle checkpointing

Using corresponding dedicated Python library, you must implement the functional utils to be able to save intermediate model checkpoints to Kaggle models during training using an external credentials file. This part implies that the notebook is executed as Kaggle kernel, and the rest of the code from the repo is cloned as a GitHub repository directly to the Kaggle session under the "instanseg" repo name. You must ensure that the model checkpoints store all the neccessary information, including logs, metrics history etc., so that we could later use them just like ordinary checkpoints to resume training properly. If needed, implement utilities for compressing all that into a single file and then decompressing it back into several ordinary files.

Checkpointing must be configured by several configurable parameters which should be passed into the `model.train()` method. Here is the list of the parameters (you should also ensure all configurable params are present in the YAML config file):

* `save_kgl_ckp` (bool, default to False) - defines whether model checkpoints are saved to Kaggle in the current experiment or not;
* `kgl_best_ckp_path` (str, default to None) - defines the path to the Kaggle checkpoint of the best model. The checkpoints are saved to the specified path. If the specified path does not exist, try creating it. If the path cannot be created, or the specified directory cannot be modified, or the value is None - raise a corresponding ValueError;
* `kgl_last_ckp_path` (str, default to None) - defines the path to the Kaggle checkpoint of the best model. The checkpoints are saved to the specified path. If the specified path does not exist, try creating it. If the path cannot be created, or the specified directory cannot be modified, or the value is None - raise a corresponding ValueError;
* `kgl_ckp_freq` (int, default to 1) - specifies how often checkpoints must be saved to Kaggle. 1 means every epoch, 2 - every second epoch, 3 - every third epoch etc;
* `kgl_creds_path` (str, default to None) - specifies the path to the JSON file with Kaggle credentials. If set to None, or the path is incorrect - raise a corresponding ValueError.

### 2. Local Kaggle notebook execution

This part defines how you should implement local Kaggle notebook execution. This funcitonality should be later used by you for testing the obtained code, as well as for me training the models locally. Here are the steps you must implement. If at any point you cannot implement a certain step because of technical or other limitations, you must create a file named `KAGGLE LIMITS REPORT.md` where you will describe what you failed to implement, why it happened, and what are the best alternatives you cna think of to overcome these difficulties.

#### 2.1. Kaggle CLI for testing automation

Create scripts like `test_on_kaggle.py` or `run_kaggle.sh` that do the following:

1. update notebook;
2. push notebook;
3. wait until completion;
4. download logs;
5. exit with success/failure.

Then, whenever you modify the training pipeline, execute this testing script. If it fails, inspect kaggle logs and fix the errors. Keep this process until everything works seamlessly.

#### 2.2. Create a fully ready notebook for Kaggle local execution

Create a Jupyter notebook for traiing an InstanSeg model locally (just from the current IDE) using Kaggle runtime capabilities. Preferably I should be able to use local codebase, local model checkpointing and locally stored data, while utilizing Kaggle cloud GPU runtime environemnt.