# REIMPLEMENTATION GUIDE

## GENERAL DESCRIPTION

Based on the existing code implementation and desired code structure, re-implement the model keeping its functionality as is (full and correct) and refactoring the code in a better way to make it more modular and comprehensible for developers.

## DETAILED TASK

You must implement the instanseg model within the `instanseg` directory. Note that it already contains some of the utilities that will be used, and you need not modify them at all - just use them, if you need.

The original implementation lives in the `instanseg_original_codebase` subdir - you must refer to it to undrstand how the model should function to ensure that you correctly transfer all nuances and aspects of the model workflow and architecture. Ensuring that the model is transfered correctly, so that it still converges during training, is your top priority and you must make everything possible to make it happen.

Also, for understanding how the code should bes structured in modules, refer to `instanseg_refactored`. You must inherit its modular structure, configs etc. to make the model comprehensible, but you must not rely on its code - the code there is incorrect. You must only inherit the structure, and the code must be inherited from the `instanseg_original_codebase`. For now, only implement the 'Instanseg_Unet' backbone option which is also present in the `instanseg_original_codebase`, and the 'centr-seed' training mode, which is exactly how the model trains in the original implementation (uses seeds to determine object centroids and tris to predict vectors pointing at the object center). Do not implement the other modules for now, since it is redundant for now.

## VALIDATE BEFORE REPORTING

You must validate that you implemented everything correctly and that the model converges during training. Here is a detailed guide on how to validate it all:

1. Run pylinter tool to ensure there are no syntax errors or other basic python errors.
2. Run the following commands to push code to git repo:

```bash
git add .
git commit -m "-"
git push -u origin main
```

3. Run the `kaggle_train.py` script or whatever you want to ensure that the script is running in the Kaggle remote env and that the model is training. Use a few samples for that to overfit the model and test if it converges at all (same samples for train and validation and testing). If the validation metrics are zeros after many epochs of training, investigate the root causes, fix them and re-run. Go through this step until the model learns to overfit the samples.
4. Run the same script using the whole dataset to see if the model can actually generalize data. If so, the task is done, and you can report it as finished.
