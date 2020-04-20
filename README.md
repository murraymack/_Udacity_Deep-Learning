# _proj-Dog-Breed-Classifier
## Dog Breed Classifier Project

This repository contains a jupyter notebook which runs through building a CNN for classifying dog breeds from a list of 133 breeds.

The original structure of the notebook is sourced from Udacity course curriculum, Deep Learning Nanodegree.

## Reference Data, etc.:

[Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)<br>
[Dog Image Data](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)<br>
[Human Image Data](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)<br>

### Hand-made Architecture
The naive architecture trained from scratch was:
>nn.Conv2d(3, 32, 3, stride=2, padding=1)<br>
>nn.Conv2d(32, 64, 3, stride=2, padding=1)<br>
>nn.Conv2d(64, 64, 3, stride=1, padding=1)<br>
        
>nn.MaxPool2d(2, 2)<br>
>nn.Dropout(p=0.25)<br>
        
>nn.Linear(3136,1028)<br>
>nn.Linear(1028,512)<br>
>nn.Linear(512,134)<br>
<br>
This architecture achieved around 10% accuracy. Not very good performance, but it is a very primitive architecture, and the intent was to see the capabilities of a simple CNN without too much resource intensive training.

### Transfer Learning Architecture - VGG 11
The chosen base architecture for a better performing classifier was VGG 11 (via [PyTorch torch.model library](https://pytorch.org/docs/stable/torchvision/models.html#id2)).
<br>
<br>
I selected VGG 11 because it was one of the smallest models on the pre-trained list, but it still showed some pretty good accuracy on ImageNet. I thought this would be suitable because if VGG 11 was able to perform on ImageNet with 1000 classifications, then dog breeds at 133 classifications should have been no problem.
<br>
<br>
I wanted to see how the generalization of the CNN portion of VGG 11 performed without tinkering with their architecture too much, so I froze the parameters in the features portion of the model but left the classifier with requiresgrade = True. To re-train the classifier, all I did was adjust the dimensions in the output layer from 1000 categories to 133.
<br>
<br>
I trained the new classifier for as long as I needed to until I saw a divergence from improvements between the test set and the validation set. Once the train set started to show improvements, but the validation set was not going any lower, I stopped training. This was the signal to me that the model was beginning to memorize the training set.
<br>
<br>
The network achieved 61% test accuracy, which was slightly above the minimum requirement. I was very pleased with this result considering my rationale along each step was geared towards a pragmatic approach of doing only what was required to complete the task. I aimed to use minimum amount of time and resources for this portion of the project, for economic purposes.
