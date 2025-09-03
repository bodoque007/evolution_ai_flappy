This was a very interesting project. I initially approached it with sheer OOP design, having many pipe and bird objects in the game. But when the birds population size was big enough to make the algorithm perform significatively (1000+), the performance drastically worsened. This happened mainly because of Python being interpreted, having dynamic typing, object overhead and heap allocation. Plus, having used Tensorflow initially for such a simple NN that does not even make use of the typical algorithms that come up whenever training a NN such as backpropagation or gradient descent.
The NN's only functionality was mutating (changing its weights and biases, simple) and holding these values, thus Tensorflow was clearly an overkill.

This forced me to (as I didn't want to change to another language) approach it with Data Oriented Design, I thought it was a good opportunity to learn such thing after having heard of it mainly in old school programming.

The then objects now became simple lists that contain information about it (e.g, for the bird, it holds its y position, y velocity and score), taking advantage of vectorized operations with numpy.

As per Tensorflow's use, I instead made the NNs by myself using simple numpy arrays (matrices so to speak), as I only needed basic matrix multiplication for the forward pass and setting up the values quickly and comfortably, something numpy excels at.

Very fun project to do in one day overall.
