% BINARYCLASSIFICATION
% This script loads the dataset, partitions it into a training and testing dataset
% with {'healthy': 0, 'ADHD': 1} class labels, trains an HMM using the training dataset,
% and reports the accuracy of the model on the test dataset.

% load data and rename variables
load('data.mat');
if exist('ObVector', 'var')
    X = ObVector;
end
if exist('labels', 'var')
    y = labels;
end
clearvars -except X y % memory management

% experiment parameters
train_percent = 0.8;
classes = {0, 1:3};      % {'healthy': 0, 'ADHD': 1}

[Xtrain, Xtest, ytrain, ytest] = partitiondata(X, y, train_percent, classes);


