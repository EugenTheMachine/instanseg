# UNET CHANGELOG

## GENERAL DESCRIPTION

Within this task you must implement several modifications of the U-net model which is directly used in the InstanSeg model. The backbones must be implemented based on the existing concolutional backbones with customly created decoders for each of them to ensure that the dimensionalities are matched and the models are working properly.

## REQUIREMENTS

### GENERAL REQUIREMENTS

Here is a general step-by-step workflow that you must follow when completing this task. You can only proceed to the next step after you have successfully completed the current step and verified that it is actually fully accomplished. You must not proceed to the next step if you have not verified that the current step is fully accomplished:

1. Analyze this spec thoroughly and carefully to make sure that you have read everything and did not miss something.
2. Analyze the existing codebase to understand the best way to implement the code changes.
3. Plan your work step-by-step to ensure that you implement every requirement into the current codebase in the most efficient way and without breaking it.
4. Implement every requirement by implementing the corresponding code changes.
5. After you have implemented in code every requirement from this file, create the corresponding tests which cover every requirement, so that you can ensure that everything works properly, as expected, and does not cause errors.
6. Run all the tests to ensure everything works as expected. If you see that some tests do not pass, correct the corresponding code parts and re-run the tests again. Repeat this process until every test passes successfully. If at some point you face the execution error which is not related to the incorrect code you produced, but related to the fact that the code executor cannot be reached, stop any actions immediately and tell me explciity that "tests cannot be executed because of some internal error". Do not proceed until I explicitly tell you to do so and await my instructions.
7. After you have executed and passed all the tests, verify the scope of the work you did with what is required in this file to ensure you actually implemented everything required. If something is missing, implement and/or test it and then verify again. Keep verifying until you ensure that you actually verified and tested everything required.
8. After successfully passing full verification from the previous step, explicitly tell me that "all the requirements have been properly implemented and tested. The task is completed".

Also note that every newly added U-net modification must be modular for better structuring and easier code comprehension. Particularly the resulting modified code should contain the following classes (name them accordingly, based on the names of the used backbone models):

* encoder (an already existing backbone from the torchvision library, but without one or several final layers, so that the latent representation would be sufficient for further segmentation);
* decoder block class (it must mirror corresponding encoder blocks from the backbone network);
* decoder class (it must consist of several sequential decoder blocks and fully mirror the backbone network);
* U-net model class (it must consist of the encoder and decoder objects; by initializing the instance of this class we should initialize the full U-net model).

Also, create the `UNet` class which will be the general class for creating U-net models of diffeent architectures. It should also be the parent class for all other U-net classes, including the already existing `InstanSeg_UNet`. It should also define the general scope of attributes and methods used for the U-net model. Also, it should be used instead of the `InstanSeg_UNet` class when creating the InstanSeg model, as well as in any other situations when the `InstanSeg_UNet` class is currently used.

### 1. Implement the `EfficientUNetB0` class

You must implement the `EfficientUNetB0` class, based on the `torchvision.models.efficientnet_b0` backbone from torchvision.

### 2. Implement the `EfficientUNetB1` class

You must implement the `EfficientUNetB1` class, based on the `torchvision.models.efficientnet_b1` backbone from torchvision.

### 3. Implement the `EfficientUNetB2` class

You must implement the `EfficientUNetB2` class, based on the `torchvision.models.efficientnet_b2` backbone from torchvision.

### 4. Implement the `EfficientUNetB3` class

You must implement the `EfficientUNetB3` class, based on the `torchvision.models.efficientnet_b3` backbone from torchvision.

### 5. Implement the `EfficientUNetV2S` class

You must implement the `EfficientUNetV2S` class, based on the `torchvision.models.efficientnet_v2_s` backbone from torchvision.

### 6. Implement the `MobileUNetV2` class

You must implement the `MobileUNetV2` class, based on the `torchvision.models.mobilenet_v2` backbone from torchvision.

### 7. Implement the `MobileUNetV3S` class

You must implement the `MobileUNetV3S` class, based on the `torchvision.models.mobilenet_v3_small` backbone from torchvision.

### 8. Implement the `MobileUNetV3L` class

You must implement the `MobileUNetV3L` class, based on the `torchvision.models.mobilenet_v3_large` backbone from torchvision.

### 9. Implement the `RegNetUNetY400mf` class

You must implement the `RegNetUNetY400mf` class, based on the `torchvision.models.regnet_y_400mf` backbone from torchvision.

### 10. Implement the `RegNetUNetY800mf` class

You must implement the `RegNetUNetY800mf` class, based on the `torchvision.models.regnet_y_800mf` backbone from torchvision.

### 11. Implement the `ResNetUNet18` class

You must implement the `ResNetUNet18` class, based on the `torchvision.models.resnet18` backbone from torchvision.

### 12. Implement the `UNet` class

You must implement the `UNet` class as the main U-net class and adapt the `InstanSeg_UNet` class accordingly, so that it would become the child class of the `UNet`. Also ensure that the `UNet` class accepts the `model_type` configurable parameter and the following values (any other values not listed below must trigger the ValueError exception):

* "instanseg_unet" - must initialize the current implementation of the `InstanSeg_UNet` class;
* "efficientunetb0" - must initialize the current implementation of the `EfficientUNetB0` class;
* "efficientunetb1" - must initialize the current implementation of the `EfficientUNetB1` class;
* "efficientunetb2" - must initialize the current implementation of the `EfficientUNetB2` class;
* "efficientunetb3" - must initialize the current implementation of the `EfficientUNetB3` class;
* "efficientunetv2s" - must initialize the current implementation of the `EfficientUNetV2S` class;
* "mobileunetv2" - must initialize the current implementation of the `MobileUNetV2` class;
* "mobileunetv3s" - must initialize the current implementation of the `MobileUNetV3S` class;
* "mobileunetv3l" - must initialize the current implementation of the `MobileUNetV3L` class;
* "regnetunety400mf" - must initialize the current implementation of the `RegNetUNetY400mf` class;
* "regnetunety800mf" - must initialize the current implementation of the `RegNetUNetY800mf` class;
* "resnetunet18" - must initialize the current implementation of the `ResNetUNet18` class.
