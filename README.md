# LSF_deep_learning

This is a video translator for french alphabet sign language.This translator is able to translate alphabet sign from A to L without the J which is in movement so not include. 

To obtain this, first, we built a train and test dataset (camera.py). For that, we decided to use the same type of image we were going to have during the video which are photos centered on the hand. Those photos are obtained by finding the hand and putting a bounding box. After obtaining 120 photos for the training group (40 per person in the group) and around 30 (10 per person) for the test set, those data were used to train the neural network. This network is a CNN based on LeNet5. The save model is then used in the video : each photo of the video is given to the model in order to obtain a prediction and the result with the probability is printed on the image.

The result is the following:


https://user-images.githubusercontent.com/103188608/215330793-78b58315-c9fc-4a05-b973-ba3a5bf4e7e2.mov


https://user-images.githubusercontent.com/103188608/215330868-8a3cbf6a-5f25-4565-a728-cce94efc2761.mov


