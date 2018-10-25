# Convolutional Autoencoders for the Cifar10 Dataset

Making an autoencoder for the MNIST dataset is almost too easy nowadays.  Even a simple 3 hidden layer network made of fully-connected layers can get good results after less than a minute of training on a CPU:

[MNIST image]

The network architecture here consisted of fully-connected layers of sizes 100, 100, 100, 784, respectively.  You might notice that these numbers weren't carefully chosen --- indeed, I've gotten similar results on networks with many fewer hidden units as well as networks with only 1 or 2 hidden layers.  

The same can't be said for the Cifar datasets.  The reason for this isn't so much the 3 color channels or the slightly larger pixel count, but rather the internal complexity of the Cifar images.  Compared to simple handwritten digits, there's much more going on, and much more to keep track of, in each Cifar image.  I decided to try a few popular architectures to see how they stacked up against each other.  Here's what I found:

 - Fully connected layers always hurt
 - Reducing the image size too many times (down to 4 x 4 x num_channels, say) destroys performance, regardless of pooling method
 - Max Pooling is OK, but strided convolutions work better
 - Batch Normalization layers do give small but noticable benefits
 - 3 convolutional layers in the encoder portion of the model seems to be the minimum number to get really good performance
 

A word before moving on:  Throughout, all activations are ReLU except for the last layer, which is sigmoid.  All training images have been normalized so their pixel values lie between 0 and 1.  All networks were trained for 10 or fewer epochs, and many were trained for only 2 epochs.  I didn't carry out any sort of hyperparameter searches, although I did increase the number of hidden units in a layer from time to time, when the network seemed to be having real trouble learning.  I found that mean squared error worked better as a loss function than categorical or binary crossentropy, so that's what I used.  Each network used the Adam optimizer available in Keras.  

Finally, my sample sizes are far too small to be sure of any of my above conclusions.  Anyone interested should play around with different sizes, numbers, and types of layers to see for themselves what works and what doesn't work.  Also, these images are so small that once an autoencoder becomes "good enough", it's hard for the human eye to see the difference between one model and another.  We could look at the loss function, but mean-squared-error leaves a lot to be desired and probably won't help us discriminate between the best models.  


# Some Poor-Performance Autoencoders












