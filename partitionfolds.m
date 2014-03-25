function [cv_inds] = partitionfolds(X, y, K, classes)
%PARTITIONFOLDS partition labelled dataset (X,y) into K folds for
%   k-fold cross-validation. Output: folds {1 x K} cell array of folds.
%   Note: folds are filled with equal distribution of class labels
%
%   Parameters:
%   X: training dataset
%   y: training labels
%   K: number of folds
%   classes: structure of classes to consider. For example, to disclude (2, impulsive)
%            types from the dataset, let classes = {0, 1, 3}; To combine classes,
%            for example to do binary ADHD classification, let classes = {0, 1:3};

[numS, numT, numF] = size(X);

class_partition = cell(length(classes), K);
for cidx = 1:length(classes)
    class_lbl = cidx-1;
    class_subset_ind = find(y == class_lbl);
    num_class_samples = length(class_subset_ind);
    % partition the class subset into k folds
    fold_size = ceil(num_class_samples/K);
    last_fold_size = num_class_samples - fold_size*(K-1);
    class_partition(cidx,:) = mat2cell(class_subset_ind, [fold_size*ones(1,K-1), last_fold_size])';
end

% concatenate class subset partitions into K folds
folds = cell(1,K);
for k = 1:K
    folds(k) = {vertcat(class_partition{:,k})};
end

% calculate indicies of train and test subjects for each cross-validation run
cv_inds = repmat(struct('train', [], 'test', []), [1, K]);
for k = 1:K
    train_folds = setdiff(1:K, k);
    test_fold = k;
    cv_inds(k).train = vertcat(folds{train_folds});
    cv_inds(k).test = folds{test_fold};
end
