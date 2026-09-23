# CRITICAL TASK

## GENERAL DESCRIPTION

Currently model training does not convrge properly after the code refactoring. You must investigate it and fix the errors that cause the issue. Find the details below.

## Details

During model training, the loss function seems to decrease for both training and validation. However, the validation metrics are all zeros, which means that something is wrong. Moreover, often no outputted pixels exceed the minimum segmentation threshold, so that it is like nothing was found at all. You must investigate it and fix the issue, so that the convergence becomes stable and real.

Possible reasons for the issue are:

1. Incorrect loss calculation.
2. Incorrect network output reading (so that we correctly compute loss, but use incorrect input data, so that the output loss also becomes incorrect).
3. Incorrect postprocessing, which "kills" network segmentation results.'
4. Incorrect quality metric calculation, so that non-zero metrics become zeros.

## Useful reeferences

We use the "center-seed" network mode, which is known to have stable convergence in normal situations, and "Instanseg_Unet" backbone, which works fine, too. You must reference the current implementation of the center-seed workmode to its original version from the `instanseg_original_codebase` dir (the backbone also exists there, too). You must determine why the model does not learn and shows poor metrics. Report to me when you are done.

NOTE that you must not alter the `instanseg_original_codebase` dir content - you can only read it for reference.

## Results validation

After you implemented some changes to the code, you must push them to github by running:

```bash
git add .
git commit -m "-"
git push -u origin main
```

Then, run the training notebook in Kaggle remote GPU env. You may use the `kaggle_train.py`, since it works great by running the latest notebook version.

## Recommended workflow

Begin from trying to overfit the model at several data samples used for both train and validation. Currently the model cannot even overfit them. Use `kaggle_overfit_notebook` for reference and improve it, if needed.

Then, when the model becomes capable of memoryzing data, start trying to train it for real, using different subsets of data for train and validaion, and see how it goes.