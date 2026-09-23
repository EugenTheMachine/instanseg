# LORA-CHANGELOG

## GENERAl DESCRIPTION

Within this task you must implement parameter-efficient fine-tuning (PEFT) options for the InstanSeg model U-net backbone. This includes implementing LoRA-C, Conv-LoRA, DoRA-C, Conv-DoRA, CP-LoRA, CP-DoRA algorithms for the layers from the U-net decoder, so that we could parameter-efficiently fine-tune the model with minimal resources and minimal amounts of data.

## REQUIREMENTS

### GENERAL REQUIREMENTS

Here is a general step-by-step workflow that you must follow when completing this task. You can only proceed to the next step after you have successfully completed the current step and verified that it is actually fully accomplished. You must not proceed to the next step if you have not verified that the current step is fully accomplished:

1. Analyze this spec thoroughly and carefully to make sure that you have read everything and did not miss something.
2. Analyze the existing codebase to understand the best way to implement the code changes.
3. Plan your work step-by-step to ensure that you implement every requirement into the current codebase in the most efficient way and without breaking it.
4. Implement every requirement by implementing the corresponding code changes. Yoy must refer to the `lora.pdf` document to see the theoretical foundation of all LoRA and DoRA variations, and you must refer to the code under the `lora-peft/` directory for the actual code implementation of the described approaches.
5. After you have implemented in code every requirement from this file, create the corresponding tests which cover every requirement, so that you can ensure that everything works properly, as expected, and does not cause errors.
6. Run all the tests to ensure everything works as expected. If you see that some tests do not pass, correct the corresponding code parts and re-run the tests again. Repeat this process until every test passes successfully. If at some point you face the execution error which is not related to the incorrect code you produced, but related to the fact that the code executor cannot be reached, stop any actions immediately and tell me explciity that "tests cannot be executed because of some internal error". Do not proceed until I explicitly tell you to do so and await my instructions.
7. After you have executed and passed all the tests, verify the scope of the work you did with what is required in this file to ensure you actually implemented everything required. If something is missing, implement and/or test it and then verify again. Keep verifying until you ensure that you actually verified and tested everything required.
8. After successfully passing full verification from the previous step, explicitly tell me that "all the requirements have been properly implemented and tested. The task is completed".

### 1. Implement LoRA for the decoder layers

Implement the possibility to perform LoRA PEFT by altering the structure of the decoder layers in the U-net part of the InstanSeg model. You must particularly implement:

* LoRA-C decomposition of the decoder layers;
* ConvLoRA decomposition of the decoder layers;
* CP-LoRA decomposition of the decoder layers;
* configurable parameter `peft` which:
  - if given value "lorac", initializes the model where the U-net part has LoRA-C configured decoder layers;
  - if given value "convlora", initializes the model where the U-net part has ConvLoRA configured decoder layers;
  - if given value "cplora", initializes the model where the U-net part has CP-LoRA configured decoder layers;
* configurable parameter `r` which defines the rank of the decomposed weight matrices for the LoRA PEFT. Set to 4 by default;
* configurable parameter `lora_alpha` which defines the Scaling factor applied to the LoRA update (α/r). Controls the contribution of the adapter. Set to 4*r by default;
* configurable parameter `lora_dropout` which defines the Dropout applied to the LoRA branch during training for regularization. Set to 0 by default;
* configurable parameter `bias` which determines whether bias parameters are trainable. If given "lora_only" value, then only LoRA matrix biases should be trainable. If given "all" value, then all bias params should be trainable. If given "none" value or None value, no bias params should be trainable. Set to "lora-only" by default.

You must decide yourself whether you will add each of these configurable parameters to the model constructor or the `model.train(...)` main training function.

### 2. Implement DoRA for the decoder layers

Similarly to what you did in part 1, implement the possibility to perform DoRA PEFT by altering the structure of the decoder layers in the U-net part of the InstanSeg model. You must particularly implement:

* DoRA-C decomposition of the decoder layers;
* ConvDoRA decomposition of the decoder layers;
* CP-DoRA decomposition of the decoder layers;
* update the behaviour of the configurable parameter `peft`, so that in addition to the earlier implemented scenarios it could also accept the following values:
  - if given value "dorac", initializes the model where the U-net part has DoRA-C configured decoder layers;
  - if given value "convdora", initializes the model where the U-net part has ConvDoRA configured decoder layers;
  - if given value "cpdora", initializes the model where the U-net part has CP-DoRA configured decoder layers.
Any other values for the `peft` param, which are not listed in part 1 or in the current part 2, must trigger the ValueError exception;
* make sure that the added DoRA implementations are fully compatible with the other configurable params implemented in part 1, including `r`, `lora_alpha`, `lora_dropout`, and `bias`.
