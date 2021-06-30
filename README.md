# Project_deep_learning
Introduction to Deep Learning: 4-part project


# purpose lab4: generate headline for an image
Pipeline:

Training pipeline: 

train two networks: 

(1) N3 = network that produces a word describing an image from lab 3

(2) N4 = LSTM network that predicts the next word given a word

Test pipeline: image -> image headline

Use N3 to produce a word W1 for given image;
Use N4 to produce two more words W2,W3 to build a headline: W2 is prediction by N4 given W1, and W3 is prediction by N4 given W2.

