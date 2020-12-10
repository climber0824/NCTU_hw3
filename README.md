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
train.py: using to train the model  
predict.py: using to predict the mask of input images  
make_submission.py: using to output .json file for submit  
config.yaml: setting for training, predict, and make submission  



**Usage:**  
1.Data preparatoin:  
Download the data from:https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK  
The dataset structure is the same as below:  
dataset  
  +- train_images  
  |  - 2007_000033.jpg  
  |  - 2007_000042.jpg  
  |  - 2007_000061.jpg  
  |  - ...  
  +- test_images  
  |  - 2007_000629.jpg  
  |  - 2007_001157.jpg  
  |  - 2007_001239.jpg  
  |  - ...  
  pascal_train.json  
  test.json  
  
2.Training:  
You can set up the related setting in config.yaml and run the following command to train:  
$ python3 train.py --config=config.yaml  
  
3.Prediction:  
You can set up the related setting in config.yaml and run the following command to predict:  
$ python3 predict.py --config=config.yaml  

4.Make Submission:  
You can set up the related setting in config.yaml and run the following command to make submission file:  
$ python3 make_submission.py --config=config.yaml
















