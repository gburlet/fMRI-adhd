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
%classes = {0, 1:3};             % {'healthy': 0, 'ADHD': 1}
classes = {0, 1, 3};
num_classes = length(classes);
num_states = 15;                 % TODO: choose this using k-fold cross validation

[Xtrain, Xtest, ytrain, ytest] = partitiondata(X, y, train_percent, classes);
[N, T, F] = size(Xtrain);

% initialize HMMs
% equiprobable state distributions
pi0 = ones(num_states, 1)/num_states;
A0 = ones(num_states)/num_states;

% initialize emission distributions using MLE from data
% Use multivariate gaussian emission distribution
% Note: each state multivariate gaussian distribution has equivalent parameters
% TODO: implement GMM emission distribution
hmms = cell(1, num_classes);
for class = 0:num_classes-1
    % gather observations for the data samples with this class label
    % preprocess the observations to be used by the EM learning function
    Obs = permute(Xtrain(ytrain == class,:,:), [1,3,2]);
    Obs = cellfun(@squeeze, num2cell(Obs, [2,3]), 'UniformOutput', false);
    
    % initialize
    Xtrain_tempavg = squeeze(mean(Xtrain(ytrain == class,:,:),2));
    mu0 = repmat(mean(Xtrain_tempavg, 1), [num_states, 1])';
    Sigma0 = repmat(cov(Xtrain_tempavg), [1,1,num_states]);
    emission0 = condGaussCpdCreate(mu0, Sigma0); 

    % train HMM
    hmms{class+1} = hmmFit(Obs, num_states, 'gauss', ...
                    'maxIter', 1000, 'convTol', 1e-7, ...
                    'nRandomRestarts', 1, 'verbose', true, ...
                    'pi0', pi0, 'trans0', A0, 'emission0', emission0);
end

% evaluate model on testing data
num_test = length(ytest);
Obs_test = permute(Xtest, [1,3,2]);
Obs_test = cellfun(@squeeze, num2cell(Obs_test, [2,3]), 'UniformOutput', false);
logp = zeros(num_test, num_classes);
for class = 1:num_classes
     logp(:,class) = hmmLogprob(hmms{class}, Obs_test);
end
[~, yhat] = max(logp, [], 2);
recall = sum(yhat-1 == ytest);
accuracy = recall / num_test;

fprintf('Classification accuracy %d/%d: %.2f%% \n', recall, num_test, accuracy);
