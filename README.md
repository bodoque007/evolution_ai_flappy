# Flappy Bird AI Evolution

## How to Run

```bash
pip install -r requirements.txt
python flappy_bird_evolution.py
```

## Project Overview

This was a very interesting project. I initially approached it with sheer OOP design, having many pipe and bird objects in the game. But when the birds population size was big enough to make the algorithm perform significatively (1000+), the performance drastically worsened.

## Performance Challenges

This happened mainly because of Python being interpreted, having dynamic typing, object overhead and heap allocation.

Plus, I was using Tensorflow initially, but this was an overkill for such a simple NN that does not even make use of the typical algorithms such as backpropagation or gradient descent.

The Neural network's only functionality was mutating (changing its weights and biases, simple), holding these values and making the forward pass to "make its choice".

## Solution: Data Oriented Design

This forced me to (as I didn't want to change to another language) approach it with Data Oriented Design, I thought it was a good opportunity to learn such thing after having heard of it mainly in old school programming.

The then objects now became simple lists that contain information about it (e.g, for the bird, it holds its y position, y velocity and score), taking advantage of vectorized operations with numpy.

## Custom Neural Network Implementation

As per Tensorflow's use, I instead made the NNs by myself using simple numpy arrays (matrices so to speak), as I only needed basic matrix multiplication for the forward pass and setting up the values quickly and comfortably, something numpy excels at.

## Conclusion

Very fun project to do in one day overall, after having gone through all stages of grief I ended up making it work.
