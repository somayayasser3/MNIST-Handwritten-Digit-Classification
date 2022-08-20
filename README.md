# MNIST-Handwritten-Digit-Classification
Using Artificial Neural Network to classify Handwritten Digits.
-First Step:
I loaded the "data.mat" file that contains the data to be classified then assign the X values to the x column in the data and Y to data['y']
-Second Step:
I loaded the "weights.mat" file that contains the initial weights for the inputs.
-Third Step:
I build the Neural Network Class with constructor and feed forward and accuracy functions:
     -constructor that holds the input, actual output, two initial weights.
     -feed forward to calculate the predicted output.
     -calculate the accuracy between the actual and predicted output.
and I also bulit the segmoid function to be used in feed forward function to calculate the predicted output.
