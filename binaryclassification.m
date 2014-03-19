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
num_states = 4;          % TODO: choose this using k-fold cross validation

[Xtrain, Xtest, ytrain, ytest] = partitiondata(X, y, train_percent, classes);
[N, T, emission.d] = size(Xtrain);

% initialize HMM
% equiprobable state distributions
pi = ones(num_states, 1)/num_states;
A = ones(num_states)/num_states;

% initialize emission distributions using MLE from data
% Use multivariate gaussian emission distribution
% TODO: implement GMM emission distribution
% Note: each state multivariate gaussian distribution has equivalent parameters
Xtrain_tempavg = squeeze(mean(X,2));
emission.mu = repmat(mean(Xtrain_tempavg, 1), [num_states, 1])';
emission.Sigma = repmat(cov(Xtrain_tempavg), [1,1,num_states]);
hmm = hmmCreate('gauss', pi, A, emission);
