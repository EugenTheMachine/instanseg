# TRAINING FORMALIZATION

## GENERAL DESCRIPTION

Create a jupyter notebook which will be pushed to kaggle for model training and then prepar python script to execute this kaggle kernel in remote kaggle gpu p100 runtime.

## STEP-BY-STEP GUIDE

This part describes what needs be accomplished and implemented as a result:

1. Kaggle training jupyter notebook which lives in the root dir of the project and which will be executed for model training. Refer to the notebook from `examples` dir to see how it can look like. Make sure to adapt it to your current codebase to avoid unexpected errors.
2. Python script for running the kaggle kernel in the remote env from the local machine, instead of using webui which is less convenient. Refer to the script from `examples` dir to see how it can look like. Make sure to adapt it to your current codebase to avoid unexpected errors.
3. Set up kaggle remote runtime where you will be running kernels from the local machine. You can do it via python scipt.
4. Use `kaggle_token.txt` file to get kaggle credentials and `kernel-metadata.json` to see how to set up your noebook and which data to use for training.
5. Use my own dataset from kaggle named `livecell-cellseg1-a172` as input data.

## VALIDATION GUIDE

Here we describe how to test if everything is implemented correctly. Follow the described workflow to ensure everything is tested. If any errors appear, fix them and begin testing again from the very first step. Here is the algorithm:

1. Run python linters to check there are no syntax errors or other basic mistakes in the code.
2. Push your code to github repo by running:
   ```bash
   git add .
   git commit -m "-"
   git push -u origin main
   ```
3. Run the python script to run the notebook and see if it is running smoothly.
4. During the run train the model as described below to see that it works correctly.

Here is how to train the model:

1. Use the `test` subfolder of the dataset as both training and validation and testing images and run training for 10 epochs to see whether there will be any progress in the loss funcitons and whether the model will start overfitting the data. Make sure the same images are used in all the subsets This way we test basic convergence capabilities of the model.
2. Use the whole dataset (both `train` and `test` subfolders) for training, and split the data into different subsets. Run training for 8 epochs to see if there will be any progress in the training.

Only after running all of that and seeing that verything works great and fine can you report the task as done.
