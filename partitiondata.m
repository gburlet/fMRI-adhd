function [Xtrain, Xtest, ytrain, ytest] = partitiondata(X, y, train_percent, classes)
%PARTITIONDATA partitions the dataset (X,y) into training and testing
% datasets. Splits the data to have equal portions of class labels in
% each set.
%
% Parameters:
%   X [NumSubjects, NumTimeSlices, NumFeatures]: fMRI scan data
%   y [NumSubjects, 1]: class labels
%   train_percent: percentage of dataset to allocate for training
%   classes: structure of classes to consider. For example, to disclude (2, impulsive)
%            types from the dataset, let classes = {0, 1, 3}; To combine classes,
%            for example to do binary ADHD classification, let classes = {0, 1:3};

[numS, numT, numF] = size(X);

% remove unwanted classes
delclass = setdiff(unique(y), unique([classes{:}]));
if length(delclass)
    X(y == delclass,:,:) = [];
    y(y == delclass) = [];
end

class_counts = zeros(1, length(classes));
for cidx = 1:length(classes)
    % relabel classes
    class_lbl = cidx-1;
    if length(classes{cidx}) > 1
        % combine classes
        y(ismember(y, classes{cidx})) = class_lbl;
    else
        y(y == classes{cidx}) = class_lbl;
    end

    class_counts(cidx) = sum(y == class_lbl);
end

% partition data
% preallocate matrices for speed
num_train_subjects = sum(ceil(train_percent .* class_counts));
num_test_subjects = numS - num_train_subjects;
Xtrain = zeros(num_train_subjects, numT, numF);
Xtest = zeros(num_test_subjects, numT, numF);
ytrain = zeros(num_train_subjects, 1);
ytest = zeros(num_test_subjects, 1);

train_samples = 0;
test_samples = 0;
for c = 0:length(classes)-1
    sample_inds = find(y == c);
    num_train = ceil(train_percent * length(sample_inds));
    num_test = length(sample_inds) - num_train;
    Xtrain(train_samples+1:train_samples + num_train,:,:) = X(sample_inds(1:num_train),:,:);
    Xtest(test_samples+1:test_samples + num_test,:,:) = X(sample_inds(num_train+1:length(sample_inds)),:,:);
    ytrain(train_samples+1:train_samples + num_train) = y(sample_inds(1:num_train));
    ytest(test_samples+1:test_samples + num_test) = y(sample_inds(num_train+1:length(sample_inds)));

    train_samples = train_samples + num_train;
    test_samples = test_samples + num_test;
end

% sanity checks
assert(num_train_subjects+num_test_subjects == numS, 'The number of train+test subjects should equal the original number of subjects');
assert(all(0:length(classes)-1 == unique([ytrain; ytest])'), 'The classes in ytrain and ytest should be the same as the classes parameter');

end
