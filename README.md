# Hand-Writing-Recognition
This repository describes my code for the recognition of 10 handwritten alphabets.

I have employed one hidden layer network with sigmoid activation and the output layer is softmax activation and the loss function is cross entropy loss. The input features is 16*8 grayscale pixel image. These pixels are converted into a row vector and each pixel value is either 0 or 1. 

The weight initialisation is two strategy. Either complete 0 or random from uniform distribution between [-0.1,0.1]. 
The weight update strategy is SGD

To run the program type the following command:

python neuralnet.py smalltrain.csv smalltest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 2 0.1

1. <train input>: path to the training input .csv file (see Section 2.1)
2. <test input>: path to the test input .csv file (see Section 2.1)
3. <train out>: path to output .labels file to which the prediction on the training data should be
written
4. <test out>: path to output .labels file to which the prediction on the test data should be written
5. <metrics out>: path of the output .txt file to which metrics such as train and test error should
be written 
6. <num epoch>: integer specifying the number of times backpropogation loops through all of the
training data (e.g., if <num epoch> equals 5, then each training example will be used in backpropogation
5 times).
7. <hidden units>: positive integer specifying the number of hidden units.
8. <init flag>: integer taking value 1 or 2 that specifies whether to use RANDOM or ZERO initialization
that is, if init_flag==1 initialize your weights randomly from a uniform distribution over the range [-0.1,0.1] (i.e. RANDOM), if init_flag==2 initialize all weights to zero (i.e. ZERO). For both settings, always initialize bias terms to zero.
9. <learning rate>: float value specifying the learning rate for SGD.


