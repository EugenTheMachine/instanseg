# REFACTORING

## GENERAL DESCRIPTION

This file contains certain requirmeents on how the code must be optimized, how it should be refactored, and which parts of the code must be deprecated. You must implement all the requirements below into the code and then check whether everything works correct or not. If there are any issues you need to fix them. The detailed instructions on how to test everything is given at the end of the file.

## FUNCTIONAL REQUIREMENTS

This section describes the requirements on how you must change the code. The requirements are grouped into several subsections. You must study and apply all the requirements below into the code. 

## DEPRECATIONS

The following parts currently implemented in the code must be deprecated, because at this point they are redundant:

1. Segmentation of any other objects than cells, such as nuclei or nucleoli. We will only perform cell segmentation, so delete all the code related to segmentation of any other objects than cells.
2. The usage of biological markers, metadata and other info than the input image itself. We will only perform cell segmentation based on the input image, so delete all the code related to the usage of biological markers, metadata and other info than the input image itself.
3. Replace custom augmentation functions with common methods from albumentations (or torchvision.transforms if albumentations is not needed, but I prefer albumentations). Then deprecate the custom augmentation functions.

## CODE RESTRUCTURING

1. Create a new file `postprocessing.py` in `instanseg/utils`. Then move all the postprocessing functions to this file. I,e. `filter_small_blobs`, `split_touching_nuclei`, `extract_cell_crops` and so on.
2. Create a new file `preprocessing.py` and move all the preprocessing funcitons into this file (at least the main methods - you can keep the utils in the corresponding utils file).
3. Make sure the `loss` dir and files in it only contain functions related to the loss function calculation. Move all other methodsnot directly related to the loss calculation elsewhere.

## TESTING INSTRUCTIONS

You must follow the instructions below when testing the code:
1. For each newly added or changed funcitonality run `pytest` and check that all tests pass. If there are any tests failing, you need to fix them. If there are no tests for the functionality you've added or changed, you need to create them (create the files in a new dir named `refactoring_tests` at the root of the project).
2. Run the tests from the `tests` folder and check that all tests pass. If there are any tests failing, you need to fix them.
3. If all the tests from steps 1 and 2 pass, you can proceed to the next step.
4. Update all the other related documentation, such as `README.md`, `CHANGELOG.md`, `TODO.md` and so on. If there are no relevant docs, create them.
5. Report the job as done and finish your work.