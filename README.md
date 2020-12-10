# NCTU_hw3
Instance segmentation using tiny dataset


**Environments:**

Python 3.6  
PyTorch 1.5


**File structure:**

+- data
|  - config.py: the setting of backbone of model and the path of dataset
|  - dataloader.py: using to download training and testing data
+- models
| - backbone.py: the backbone(resnet101+FPN) of this model
| - box_utils.py: using to calculate IOU of bbox or mask
| - detection.py: using to calculate NMS
| - interpolate.py: class to interpolate
| - model.py: the whole predict model
| - multibox_loss.py: using to calculate the bbox loss, label loss, mask loss, and segmentation loss
| - output_utils.py: post processing for the output of model
+- utils
|  - augmentation.py: function for data augmentation
|  - functions.py: using to print processing bar
|  - timer.py: using to print training time
- train.py: using to train the model
- predict.py: using to predict the mask of input images
- make_submission.py: using to output .json file for submit
- config.yaml: setting for training, predict, and make submission




