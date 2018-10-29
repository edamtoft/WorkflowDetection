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