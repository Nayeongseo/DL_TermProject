# DL_TermProject

## Project Duration
2024.11.06 - 2024.12.11

## CNN Architectures Used
- AlexNet
- VGGNet
- ResNet
- GoogLeNet

## Kaggle Competition
[State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data)

### Dataset Description
 - Driver Images: each taken in a car with a driver doing something in the car
 - Goal is to predict the likelihood of what the driver is doing in each picture. 

### Classes
 - c0: safe driving
 - c1: texting - right
 - c2: talking on the phone - right
 - c3: texting - left
 - c4: talking on the phone - left
 - c5: operating the radio
 - c6: drinking
 - c7: reaching behind
 - c8: hair and makeup
 - c9: talking to passenger

### File descriptions
- imgs.zip : zipped folder of all (train/test) images
 - sample_submission.csv : a sample submission file in the correct format
 - driver_imgs_list.csv : a list of training images, their subject (driver) id, and class id
