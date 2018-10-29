# WorkflowDetection
Leveraging Deep Learning and Tensorflow.js to Learn Workflows

## About

This example uses [Tensorflow.js](https://js.tensorflow.org) to create and train a deep neural network
on the client side to detect workflow based on navigation through a site. In general, for a web application,
there will a few unique "pathways" through the application that a user will repeatedly take. The customized
neural network will learn the pathways that an individual takes through an application and start to predict
your next move. This could be used to populate "quick links" or of some kind, or even automate actions
based on certainty.

## Usage

To use, choose a few patterns
and click around following those patterns for about 100 clicks. Then click the "Learn" button. This will
build and train the model based on the first usage. A current limitation of the system is that the model must
manually be created before it will start predictions. After that point, it will periodically re-train to
improve predictions and learn new patterns.

## Live Demo

https://edamtoft.github.io/WorkflowDetection/index.html

## Neural Network Architecture

This sample uses a simple deep neural network and trains the model as a state machine based off the last 5 actions.
This keeps the model simple, but also sophisticated enough to do a reasonably good job at detecting simple patterns.

The network takes a 5x8 input (5 most recent pages * 8 unique pages represented as one-hot tensors) and generates a
flat 8 wide output which represents probabilities of each individual page. There are two hidden layers with 10 and 
15 neurons respectively. This was a mostly arbitrary choice that seemed to provide relatively good results.

A recurrent neural network using LSTM nodes may also do a good job and be better fitted to this problem, so that might
be worth investigating as an alternative.