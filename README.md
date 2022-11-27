# Using Depth-Camera Sensors to Estimate Body Weight in Dairy Cows

## Abstract:
Sensing technologies are used in precision livestock farming to capture morphometric changes in animal growth and body composition dynamics. Computer vision technologies, are critical for speeding up phenotyping efforts by providing non-intrusive structural assessments of animals with high temporal and spatial resolution. Our aim was to create a computer vision system that uses depth sensor cameras to automatically estimate body weight of dairy cattle. A total of 12 Holstein animals from the Dairy Complex on Kentland farm (Virginia Tech, Blacksburg, VA) were used. Data collection occurred after cows exit the milking parlor from the 12AM and 12 PM milking sessions, and repeated every day for consecutive 30 days. A depth sensor camera connected to a laptop captured top-view RGB and depth images of cows walking below the camera in an unconstrained manner after exiting the milking parlor. OpenCV was used in Python to perfom image segmentation and thresholding to extract image features. From the image features, we determined the cow’s length, width, height and volume. Length, width, height and volume were used as predictors to build a regression model and then predict the dairy cow’s body weight. To test your model performance, we performed cross-validation designs. Mean square error and pearson correlation coefficients between training and testing were used to evaluate methods. Correlation between image features and body weight ranged from 0.66 to 0.92, and cross-validation designs correlation ranged from 0.89-0.95.

## Relative codes:
- [test.py]
- [BodyWeightCorr.Rmd]
- [Prediction.Rmd]
