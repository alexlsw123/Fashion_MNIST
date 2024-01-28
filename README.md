# Fashion_MNIST

## Data

The data can be found from kaggle:
`https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download`

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

## Task

My job is to find optimal test and train accuracy with the CNN model.

## Result

When I created the model, I defined the convolutional block using `Conv2D`. In `Conv2D` for three times with different values of filter to learn more complex and diverse features from the input for better model performance. For all three convolution block, I used the `He normal` initializer for the convolutional kernels that helps in efficient weight initialization, and used Batch normalization to normalize the inputs to a layer, stabilizing and accelerating the training process. Next, I added Max pooling to reduce the spatial dimensions of the feature maps, which captures the most important information. I next added dropout layer to prevent overfitting which randomly drops the input units during training (I used 0.5 for the value since I thought it was a reasonable amount). I then flattened the layers which converts the 3D feature maps into a 1D vector, preparing them for input to the dense layers, and added 2048 then 1028 neurons with ReLU activation (also included batch normalization and dropouts). The final dense layers further reduce the dimensionality, with the last layer having 10 neurons corresponding to the 10 classes in Fashion MNIST (used softmax activation function since it is a multi-class classification).

For compiling the model, I tried different optimizers such as adam, adamax, adagrad, AdamW, and SGD, but I finalized it with adam performed the best out of all of them (AdamW achieved at the point of 94.8% test accuracy, but it seemed like it was overfitting as the train accuracy was around 98-99%).

After compiling, I added a line before fitting the model that model automatically drops the learn rate when the model stops improving by using `ReduceLROnPlateau`.  
I fitted the model with the batch_size of 32 and 50 epochs. I acknowledge that 50 epochs is too much, so I tried to use `EarlyStopping` initially; however, I decided to remove it as I was going to plot the output later on and wanted to clearly show the learn rate converges at some point.

I next evaluated the model for the last result of the test accuracy and achieved **~94.6%** test accuracy (result attached). 

Finally, I plotted the output of fitting the model for both train and test image for comparison.

