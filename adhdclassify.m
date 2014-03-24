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
classes = {0, 1, 3};             % {'healthy': 0, 'ADHD-i': 1, 'ADHD-c': 2};
num_classes = length(classes);
num_states = 15;                 % TODO: choose this using k-fold cross validation
num_gmix = 5;
emission_type = 'gmm';
%emission_type = 'mvg';

[Xtrain, Xtest, ytrain, ytest] = partitiondata(X, y, train_percent, classes);
[N, T, F] = size(Xtrain);

hmms = cell(1, num_classes);
for class = 0:num_classes-1
    % gather observations for the data samples with this class label
    if strcmp(emission_type, 'mvg')
        % train hmm with multivariate Gaussian emission distribution
        hmms{class+1} = hmmtrain_mvg(Xtrain(ytrain == class,:,:), num_states, true);
    elseif strcmp(emission_type, 'gmm')
        % train hmm with Gaussian mixture model emission distribution
        hmms{class+1} = hmmtrain_gmm(Xtrain(ytrain == class,:,:), num_states, num_gmix, true);
    end
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
