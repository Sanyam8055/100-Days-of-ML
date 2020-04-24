# 100-Days-of-ML


## Day 1 (25-03-20) : Binary Classification
- Implement a Deep Neural Network for Classification of Cats and Dogs.
- Tweaked the model by feature scaling and hyperparameter tuning.
- Minimized overfitting by adding image augmentation through Image Data Generator.
- Achieved an accuracy of 90 percent without using any dense layers in the network.
![](Images/c.png) 
- Model Link ~ https://colab.research.google.com/github/Sanyam8055/100-Days-of-ML/blob/master/catsvsdogs.ipynb
## Day 2 (26-03-20) : Multi-class Classification
- Implemented a Resnet20 for Multi-class classification on CIFAR100 dataset.
- Tweaked the Learning rate by applying LR reducer and LR scheduler.
- Model Link ~ https://colab.research.google.com/github/Sanyam8055/100-Days-of-ML/blob/master/Resnet20.ipynb
## Day 3 (27-03-20) : Neural Style Transfer 
- Runs on custom Image with an custom filter
- The model is uses characters of one Image as a filter 
- Tweaked the loss function to compute better results
<img src="Images/a.png" width="50%" height="40%">

- Model Link ~ https://colab.research.google.com/drive/12cuuIp1JrTiuqhqS2YY6eRwZzClyN1Bg
## Day 4 (28-03-20) : Binary Person Classifier 
- Extracts important featuers from different datasets
- Identifies on a large variety of user-defined dataset
![](Images/abc.png)
- Model Link ~https://colab.research.google.com/drive/12cuuIp1JrTiuqhqS2YY6eRwZzClyN1Bg
## Day 5 (29-03-20) : Mathematics for ML
- Studied Gaussien Naive Bayes theorem 
- Some concepts of sampling including Random Sampling, Systematic Sampling and Stratified Sampling.
- Statistic Strategy including Descriptive and Inferential.
- Link - https://machinelearningmastery.com/naive-bayes-for-machine-learning/
## Day 6 (30-03-20) : Keras Implementation of Custom Layer
- Custom layer with lecum_uniform and selu activation
- Specifically for SNN 
- Uses recursive loss to evaluate loss that going through the layer.
- Link for Layer ~ https://github.com/Sanyam8055/100-Days-of-ML/blob/master/Customdenselayer.py
## Day 7 (31-03-20) : Custom model for cifar10
- Model achieves an accuracy of 83 percent under 50 epochs
- Model is built up of convolutional layers with any involvement of dense layers.
![](Images/b.png)
- Model Link ~ https://colab.research.google.com/drive/1TJml50aCS-wSTebExg-TvgOXFWOhHP0z
## Day 8 (01-04-20) : Music Generation using RNN
- Preprocessed the songs into vectorized text for the model
- Build a Recurrent neural network with LSTM and dense 
- Customized the loss function for the model
- Custom Song Link ~ https://drive.google.com/file/d/1NpjvOh9Kk9JqfEcO_hYsiSvGyxSPb0Rw/view?usp=sharing
## Day 9 (02-04-20) : Customized Music Generation 
- Customized the optimizer by hyperparameter tunning followed by tweaking the tape gradients 
- Tweaked the batch size,  changing the starting_string and altering the rnn_units
- Reduced the loss from scalar 4.4 to 0.5
![](Images/mg.png)
- Model Link ~ https://colab.research.google.com/github/Sanyam8055/100-Days-of-ML/blob/master/Music_Generator.ipynb
## Day 10 (03-04-20) : CNN on MNIST dataset
- Implemented a convolution neural network on MNIST handwritting dataset
- Using tape gradients concluded with the backpropogation
- Sidewise compared the cnn_model with full connected model
![](Images/mnist.png)
- Model Link ~ https://colab.research.google.com/github/Sanyam8055/100-Days-of-ML/blob/master/MNIST.ipynb
## Day 11 (04-04-20) : Variational Autoencoder 
- Build a facial detection model that learns form latent variables underlying face image dataset
- Adaptively re-sample the training data
- Mitigating any biases that may be present in order to train a debiased model
![](Images/vae.png)
## Day 12 (05-04-20) : Optimized Variational Autoencoder 
- Tweaked the model while reducing the learning rate.
- Trained the model for longer num_cycles
- Better predictions on test dataset with optimum probality without any bias 
![](Images/cvae.png)
- Model Link ~ https://github.com/Sanyam8055/100-Days-of-ML/blob/master/Customized_VAE.ipynb
## Day 13 (06-04-20) : Cartpole through Reinforcement Learning 
-  The main objective of cartpole is to balance a rod kept on a subject while completely moving the surface within 2.4 units from the centre.
- Implemented MIT 6.S191 Lab 3 Cartpole with total reward of 200 under 1000 iterations
![](Images/cp.png)
- Model Link ~ https://colab.research.google.com/github/aamini/introtodeeplearning/blob/master/lab3/RL.ipynb
## Day 14 (07-04-20) : Pong with AI
- Implemented a Reinforcement learning AI which plays PONG and beats the CPU
- Pong being one the most complex games the model is trained over 2000 iterations and effective reward system.
- Training took 6 hours on google colab. 
- Further optimization required!
![](Images/pa.png)
## Day 15 (08-04-20) : Enchanced Pong 
- Trained Pong over local setup which includes setting up tf GPU on NVDIA 1060ti 6GB.
- Trained for over 10k iterations and beats the cpu with ease.
![](Images/ca.gif)
- Model Link ~ https://github.com/Sanyam8055/100-Days-of-ML/blob/master/untitled1.py
## Day 16 (09-04-20) : Papers and Papers
- Read about CGAN and its effectiveness on Face aging models.
- Read about CartoonGAN: Generative Adversarial Networks for Photo Cartoonization.
- Reads about Autoencoders and theirs differences with VAE.
## Day 17 (10-04-20) :Conditional Generative Adversarial Network
- Trained a CGAN for MNIST for 40k iterations
- Archieved discriminator accuracy of 72% and reduced Generator accuracy to 24%
- cgan_mnist  labels for generated images:  [0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]
![](Images/mm.png)
## Day 18 (11-04-20) : Custom CGAN 
- Customized model with better results
- Improved accuracy with hyper parameter tuning and increased training iterations
- Experimented with the weights 
[discriminator loss: 0.461816, acc: 0.734375] [adversarial loss: 1.522949, acc: 0.375000]
[discriminator loss: 0.475403, acc: 0.796875] [adversarial loss: 1.922817, acc: 0.156250]
[discriminator loss: 0.500307, acc: 0.765625] [adversarial loss: 2.060154, acc: 0.156250]
[discriminator loss: 0.544482, acc: 0.750000] [adversarial loss: 1.687811, acc: 0.187500]
- Model Link ~ https://colab.research.google.com/github/Sanyam8055/100-Days-of-ML/blob/master/Custom_cgan.ipynb 
## Day 19 (12-04-20) : Basic Flutter 
- Completed 6 sections of appbrewery course on flutter
- Implemented basic card app with proper User interface
- Added multiple attributes and adjusted their display.
## Day 20 (13-04-20) : Flutter Realtime Object Detection using tflite
- Flutter app for object detection through camera with accurate estimate of object and their pose.
- Works with models such as ssd-mobilenet, yolo, mobilenet and poseNet.
- Completed 2 sections of appbrewery course on flutter
- Got some really interesting results.
## Day 21 (14-04-20) : Flutter Dice App
- Flutter app for dice using Flatbottons and generating random values with random library of dart.
- User friendly and can be integrated in many games.
- Completed 2 sections of appbrewery course on flutter
## Day 22 (15-04-20) : Mathematics for ML
- Revised some concepts of numpy, pandas with Statistics.
- Built an basic OCR for Image Detection which is going to be used for Document Detection.
- Went of some major concepts of VAE in Deep learning through CMU Introduction to Deep Learning 12.
## Day 23 (16-04-20) :Camera/Gallery Plugin Flutter
- Built a flutter app that uses camera or gallery image as input.
- Displays the selected real time or previously clicked image on the home page.
- Further going to add some filters on the image using flutter ML toolkit.
## Day 24 (17-04-20) : Revision of Machine Learning 
- Completed one assignment of SHALA2020 with begineers testing code of numpy,pandas and matplot.
- Filled the missing values with mean of the columns and normalized some columns as required.
- Performs some operations and displayed the results of sin functions using matplot.pyplot
## Day 25 (18-04-20) : Logistic Regression with a Neural Network mindset
- Completed 2 Weeks of Neural Networks and Deep Learnign course by Deep Learning.AI
- Completed the first programming assignment of Implementing A neural network for Logistic Regression on a Cat dataset.
- Improved the model and successfully got an accuracy of 80%
## Day 26 (19-04-20) : SHALA2020 ASST on Data Science
- Completed one assignment of SHALA2020 with begineers testing code of pandas and numpy on movie dataset and train dataset of people who left an org after undergoing changes.
- Done all the Data Preprosseing and Visualization on both the datasets.
## Day 27 (20-04-20) : Planer Data Classification
- Completed week 3rd of Deep learning.AI with a programming assignment on Forward and Backward Propogation of Neutral Network
- Implemented a Neural Network with Relu activation followed by tanh and signmoid on flower dataset.  
## Day 28 (21-04-20) : Deep Neural Network Application
- Completed Neural Network and Deep Learning course of deeplearning.ai. 
- Implemented grader functions such as linear_forward_activation, linear_backward_activations to update the parameters after every desent.
- Used these fucntions to train a Binary Classifier of Cats in separate notebook.
## Day 29 (22-04-20) : Data Visualisations
- Learned implementation of histogram, stacked graphs, heatmap.
- Implemented a mask on the heatmap to remove duplicate values on the heatmap. 
- Implemented the probability distributions in seaborn and matplotlib
- Completed the assignment exercise for visualisation given by the IITB course
## Day 30 (23-04-20) : Improving Deep Neural Networks
-  Implemented Initialization, Regularization and Gradient Checking Checking notebooks.
- Worked on Random Initialization, He Initialization and Zeros Initialzation and found He Initialization fetching the best Results.  
- Tried different Regularizations Techniques and concluded with Dropout as one  of the best Technique fetching 98% Test Accuracy.
- Performed and Verified Gradients Testing while performing forward and backward propogation
- Completed 3 programming asst in Deeplearning.ai course. 
## Day 31 (24-04-20) : Optimization Methods
-  Tried Gradient Desent, Momentum and Adam Optimization aproaches.
- Implemented all of them on minibatches to get better results.
- Adam Fetched 94% Accuracy wheras Momentum and Gradient Desecent worked out with ~80% accuracy.
