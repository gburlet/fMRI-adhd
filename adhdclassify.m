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
%classes = {0, 1:3};                % {'healthy': 0, 'ADHD': 1}
classes = {0, 1, 3};                % {'healthy': 0, 'ADHD-i': 1, 'ADHD-c': 2};
num_classes = length(classes);
states = [4, 8, 12, 16, 20];        % vector of state parameters to try
num_gmix = 5;
K = 5;                              % number of cross validation folds
emission_type = 'gmm';
%emission_type = 'mvg';              

[X, y] = preprocess_labels(X, y, classes);
[cv_inds] = partitionfolds(X, y, 5, classes);

% for each state parameter to try
states_acc = zeros(1,length(states));
for sind = 1:length(states)
    % for each fold
    cv_acc = zeros(1,K);
    fprintf('Model evaluation: number of states = %d \n\t', states(sind));
    for k = 1:K
        fprintf('[fold %d] ', k);
        hmms = cell(1, num_classes);
        % for each class label
        for class = 0:num_classes-1
            Xtrain = X(cv_inds(k).train(find(y(cv_inds(k).train) == class)),:,:); 
            if strcmp(emission_type, 'mvg')
                % train hmm with multivariate Gaussian emission distribution
                hmms{class+1} = hmmtrain_mvg(Xtrain, states(sind), true, 1);
            elseif strcmp(emission_type, 'gmm')
                % train hmm with Gaussian mixture model emission distribution
                hmms{class+1} = hmmtrain_gmm(Xtrain, states(sind), num_gmix, true, 1);
            end
        end

        % evaluate model on testing data
        Xtest = X(cv_inds(k).test,:,:);
        ytest = y(cv_inds(k).test);
        num_test = length(ytest);
        Obs_test = permute(Xtest, [1,3,2]);
        Obs_test = cellfun(@squeeze, num2cell(Obs_test, [2,3]), 'UniformOutput', false);
        logp = zeros(num_test, num_classes);
        for class = 1:num_classes
             logp(:,class) = hmmLogprob(hmms{class}, Obs_test);
        end
        [~, yhat] = max(logp, [], 2);

        recall = sum(yhat-1 == ytest);
        cv_acc(k) = recall / num_test;
    end
    states_acc(sind) = mean(cv_acc);
    fprintf('\n\tClassification accuracy %.2f%% \n', states_acc(sind)*100);
end

states_acc
