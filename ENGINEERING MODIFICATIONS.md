# ENGINEERING MODIFICATIONS

## BRIEF DESCRIPTION

This file contains several technical requirements that the repository code must meet. You must thoroughly analyze the file, then analyze the existing code, compare them and infer which requirements are not met. Next, you must implement the missing requirements by modifying the code in the repository. Finally, run some tests for the newly implemented functionality (without testing the already existing parts) to see whether everything works fine without producing errors.

**NOTE THAT YOU MUST IGNORE ANY INFORMATION FROM OTHER .md FILES IN THIS DIRECTORY, EXCEPT FOR THE FILES DESCRIBING WHAT SHOULD BE DONE IN THE FUTURE, AND RELY SOLELY ON THE ACTUAL CODEBASE TO UNDERSTAND HOW THE CODE LOOKS LIKE NOW!!!!!!**

## WHAT SHOULD BE DONE

Here we give several parts each corresponding to a specific requirement which must be met in the codebase.

### 1. Changed input data format expected

Currently the model tends to be working with .pth datasets of microimages. Instead, you must implement a new structure of the dataset, so that the expected input data would be in the following format:
```text
dataset-folder/
|---train/
|     |---images/
|     |     |---image_0.tif
|     |     |---image_1.tif
|     |     |---...
|     |---masks/
|     |     |---mask_0.png
|     |     |---mask_1.png
|     |     |---...
|---test
|---val
```

Here the `train`, `test` and `val` subfolders contain different subsets of data (note that if there is no `val` subfolder, we should take the 10% of data from the `train` folder for this). The `images` subfolder contains TIFF brightfield microimages of cells, and the `masks` subfolder contains the 16-bit one-channel PNG masks of cell instances in index-map format, where every pixels of each cell instance are represented by the same distinct int values, and background is represented by 0-values, so that the rabge of values goes from 0 (background) to N (being number of instances).

You must make sure to change the current code, so that it correctly work with this input dataset format.

### 2. Efficient memory operations

You must ensure that the pytorch dataset and dataloader class instances are using memory-efficient operations for data reading/writing. You must make sure that the operations are as efficient as possible, so that as little time as possible is spent on them. Also, make sure to run data reading and writing in parallel to the main training or inference loop, so that the training time would be reduced.

### 3. Reproducibility

You must fix ALL the seeds for ALL the random operations in the script with the value from the config file, so that the training pipeline becomes fully reproducible. This means that if the training pipeline is repeated two times with the same config params, the same dataset and the same augmentation pipeline, it must produce two models with exactly the same weights both times. Moreover, the history of train and validation metrics also must match, because every step from attempt 1 must stay the same in attempt 2. You must test this particular funcitonality by running a pseudo-training for 2 epochs on the dataset specified in the next chapter (use image size of 128 px in order no to run out of RAM), then compare the results and tell me whether they match or not. For this, remember to fix all random seeds, values etc., including those used for shuffling data in datasets and dataloaders.

### 4. Add code quality checks

Implement code quality checks using linters and formatters like mypy, ruff, black, and isort. You should add any other modules for code quality checks that you think are useful, and you can remove any of the listed modules if you find them bad. Also, compose the file named 'README technical.md' where you briefly list all used modules with a 1-2 sentence description for each of the telling people what each module does, and then give some CMD prompt commands to perform code quality checks and/or code fixes.

### 5. Pipeline simplifications

You must make sure that the following requirements are met:

* Make sure that the full end-to-end pipeline for training can be initiated by a single function called "train(...)" which you could call to train the whole model and obtain the artifacts. The function must be a method of the model instance, so that we could call it like this:
* 
  ```python
  # training a model from scratch
  from instanseg import InstanSegModel

  model = InstanSegModel()  # to init model with random weights
  model.train(epochs=12, imgsz=128)  # training with two overridden params and others at default values
  ```

  ```python
  # resuming model training
  from instanseg import InstanSegModel

  model = InstanSegModel("checkpoint.pth")  # to load a pre-trained model
  model.train(epochs=12, imgsz=128, resume=True)  # resuming model training; note that epochs secifies total number of epochs, so here if the model is already trained for 5 epochs, it must not train longer than 7 epochs. Also, patience param should also be adjusted accordingly, based on the previous training history
  ```

The function also must expect config params listed in the config.yaml file, so that we could overwrite the default values from the file by passing the corresponding arg values directly to the function. Make sure to change the config.yaml file artifact saving accordingly, so that it is not just copying of default files, but creating of the files with actual config values.
* Make sure that evaluation pipeline (for validation or testing) can be called by calling one single function 'eval(...)' which is also a method for the InstanSeg model instance. It must meet all other requirements which are written for the 'train(...)' function and which are applicable for the evaluation pipeline.

### 6. Create a training notebook

You must create a Jupyter notebook which I could upload to Kaggle and then use to train models using Kaggle's GPUs. The Notebook must contain only code cells, so if you need to leave comments, keep them short and write them in the code cells as code comments.
he notebook must include the following code parts:

1. Repository cloning from git (keep the git link as 'LINK' for now - I will replace it later myself).
2. Session environment setup using poetry and the pyproject.toml file (make sure to install poetry via pip first).
3. Import the everything needed for the further steps.
4. Specify the path to the dataset folder as 'DATA_DIR' (I will replace it later myself).
5. Init the InstanSeg model and train it from scratch for 2 epochs using 10% of the training data, imgsz of 128 and all other default params.
6. In another cell, load a pre-trained model from a checkpoint and resume the training for 2 epochs starting from the checkpoint. Specify the path to the model as 'MODEL_PATH' (I will replace it later myself).

You should not test the notebook - I will test it in Kaggle environment myself.

You can refer to `training_kaggle.ipynb` to see how the notebook might look like and compose something like this.

### 7. Implement training changes described in `TRAIN MODIFICATIONS.md`

Implement everything descibed in the specified file and test it accordingly to ensure eveything works properly.

## GENERAL REQUIREMENTS FOR TESTING

### Testing pipeline

After implementing the changes, use the following strategy to ensure everyhting is correct:
1. Run some Python linters and other syntax check tools to automatically detectsome syntax inconsistencies and fix the at-once.
2. After passing all checks from step 1, push everything to git to ensure the changes are available for cloning - use this bash code snippet:
```bash
git add .
git commit -m "-"
git push -u origin main
```
Then, use your Kaggle MCP server tools to run the full model training pipeline, including environemnt setup, data preprocessing, model training and testing, in kaggle remote P100 GPU runtime. Use the `eugenfromkharkov/overfit-dataset` dataset for testing purposes, unless another dataset is specified for testing within a particular task description. Make sure to compose some thorough tests for ensuring that everything works as needed (e.g., when testing reproducibility).
3. Monitor the run status every 60 seconds. If there are any errors, correct them, and then re-run the full testing pipeline from the very beginning to ensure there are no more errors.
4. After successful run and test passes, report the task as done.

### USEFUL TOOLS AND INFO

1. You can refer to `kaggle_train.py` to see how to run kaggle notebooks in remote GPU runtime from local machine. Make sure to set up the correct runtime (with P100 GPU) and to have pushed the kaggle notebook the first place.
2. Refer to `kaggle_token.txt` to see the kaggle token and use it for kaggle communication and kaggle MCP server tools.
