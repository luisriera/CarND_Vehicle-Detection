# CarND_Vehicle-Detection
The Vehicle-Detection is the last project for the **Udacity Self Driven Car Nano Degree** Term 1.


# Vehicle Detection Project

---
This is the last project for the **Udacity Self Driven Car Nano Degree** Term 1, including the images from Udacity `object-dataset`. 

---
The goals for this project as set by the Udacity team are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[Vehicle_imgs]: ./writeup_img/Vehicle_imgs.png
[Non-Vehicle_imgs]: ./writeup_img/Non-Vehicle_imgs.png
[HOG_Vehicle]: ./writeup_img/HOG_Vehicle.png
[HOG_Non-Vehicle]: ./writeup_img/HOG_Non-Vehicle.png
[img_with_grid]: ./writeup_img/img_with_grid.png
[windows_overlap]: ./writeup_img/windows_overlap.png
[heatmap]: ./writeup_img/heatmap.png
[out video]: ./videos/out_project_video.mp4
[anmated_boxes]: ./videos/anmated_boxes.gif




---

I started the project by exploring and displaying few random `vehicle` and `non-vehicle` images, as show below.

**Vehicles**

![Vehicle_imgs]


**Non-Vehicles**

![Non-Vehicle_imgs]

## Histogram of Oriented Gradients (HOG)



The code used to extract the HOG features from training data can be seen in cells 6 to 8 of the Jupyter notebook. All HOG features were extracted using the hog function from skimage.  The parameters used were those recommended in the Udacity course and on the Forum, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`.  


During the process, I explored different color channels by gathering random images from the Vehicle and Non-Vehicle classes as shown below.
**Vehicles**
![HOG_Vehicle]

**Non-Vehicles**
![HOG_Non-Vehicle]

I also experimented with the parameters in the table below.  From it, I decided to use the parameters shown in row 9 for the subsequent operates as it has one of the best accuracy but still have a smaller Feature Size.
 
 |Id|Color|Spatial|Bins|Orient|Pixels/Cell|Cells/Block|Feature Size|Training Time|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|RGB|None|None|5|(8, 8)|(2, 2)|2940|80.76s|95.65%|
|2|RGB|None|None|7|(8, 8)|(2, 2)|4116|124.86s|96.36%|
|3|RGB|(32, 32)|32|9|(8, 8)|(2, 2)|8460|215.5s|98.69%|
|4|HLS|(32, 32)|32|9|(8, 8)|(2, 2)|8460|180.51s|99.32%|
|5|YUV|(32, 32)|32|9|(8, 8)|(2, 2)|8460|172.4s|99.21%|
|6|YCrCb|None|None|5|(8, 8)|(2, 2)|2940|73.3s|98.35%|
|7|YCrCb|None|None|7|(8, 8)|(2, 2)|4116|98.15s|98.27%|
|8|YCrCb|None|None|9|(8, 8)|(2, 2)|5292|123.32s|98.72%|
|**9**|**YCrCb**|**(16, 16)**|**16**|**9**|**(8, 8)**|**(2, 2)**|**6108**|**162.61s**|**99.32%**|
|10|YCrCb|(32, 32)|32|9|(8, 8)|(2, 2)|8460|165.73s|99.25%|
|11|YCrCb|(64, 64)|32|9|(8, 8)|(2, 2)|17676|204.76s|99.32%|


**Training The Model**
After choosing which parameters to use, the Support Vector Machine (SVM) linear training was implemented using spatially binned color, and historgram of colors as features.

First, all the features were extracted of the vehicle and non-vehicles from the given images in cells 9 to 13. 
To accomplish this, the images were transformed into YCrCb color space before the feature were extracted. 
Once all features were extracted, the image were scaled using a StandardScaler from the sklearn library.

The accuracy obtained by the `LinearSVC` was 99.17%.


---
## Sliding Window Search

**Implementation**
The sliding window search functionality is in cells 19 to 25.  The main Sliding Windows Search function is `find_cars`, other functions work as helper or to displays the user information related to the images or the implemented model.  The `find_cars` function takes up to five parameters; however, only the `img` and `sacle_windows` parameters are needed to use the function.  The other three parameters have defaults values that can be change by the users as needed.

The `find_cars` function takes in an image in the BGR format, which is crops to the specify dimensions, converted to `YCrCb` color space, and later scale by a the specify factor. After this initial preparation is completed, the function carryout the following sequence of process HOG, Spatial Features, Color Histogram Features, and Hot Features.  All these features are then concatenated to generate and output a `heatmap`.


**Optimization**

* The Vehicle search was limited to the part of the image where there were most likely to appear in the images. 
* Three different windows scale search were used in the pipeline aiming to detect images of the different size and on the different location.
* Small scale search windows were conducted in the middle of the image, where vehicles tend to be far in the horizontal and therefore small.
* Thresholds on heatmaps to reduce false detections.


**Examples of test images**

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  

In the following sequence of images, it can be appreciated the resulting output of the implemented pipeline.  
![heatmap]




---

### Video Implementation

![anmated_boxes]

The [out video] is a combination of the lane and vehicle detection project.


---

### Discussion


* I have explored the Udacity ` object-dataset.tar` images, but decided not to use them mainly because lack of time to generate a set of Vehicle and set of Non-Vehicles of similar number of samples in each of them.  However, I believe it should have been very handy for the augmentation of the data.

* I believe that there is an overfitting problem as the model accuracy is considerable high but still misclassified the Non-Vehicles in the output video. 
 

  

### Credits
Most of the code used for this project came out of [Frank Kanis](https://github.com/frankkanis/CarND-Vehicle-Detection) project.  I included and make some code changes to the original code from F. Kanis.
