# Deep learning project
Introduction to Deep Learning: 4-part project

# Purpose part 1: preprocess and visualize English sentences
Pipeline:

text -> preprocessed text -> 1-hot representation of words -> frequencies -> visualization

# Purpose part 2: find the sentence most similar to given query
Pipeline:

query -> preprocess query -> compute query representation
sentences -> preprocess sentences -> compute sentence representations

# Purpose part 3: classification of images with text labels
Pipeline:

Training: CIFAR-100 dataset and its labels -> CNN -> classification model

Test: image ->preprocess image -> CNN -> get word vector of a label -> find 3 most relevant text labels

# purpose part 4: generate headline for an image
Pipeline:

Training pipeline: 

train two networks: 

(1) N3 = network that produces a word describing an image from lab 3

(2) N4 = LSTM network that predicts the next word given a word

Test pipeline: image -> image headline

Use N3 to produce a word W1 for given image;
Use N4 to produce two more words W2,W3 to build a headline: W2 is prediction by N4 given W1, and W3 is prediction by N4 given W2.

