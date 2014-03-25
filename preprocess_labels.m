function [X, y] = preprocesslabels(X, y, classes)
%PREPROCESSLABELS remove unwanted class labels and relabel classes according to the
%   groups given in the parameter classes.
%
%   Parameters:
%   X: training dataset
%   y: class labels
%   classes: structure of classes to consider. For example, to disclude (2, impulsive)
%            types from the dataset, let classes = {0, 1, 3}; To combine classes,
%            for example to do binary ADHD classification, let classes = {0, 1:3};

% remove unwanted classes
delclass = setdiff(unique(y), unique([classes{:}]));
if length(delclass)
    X(y == delclass,:,:) = [];
    y(y == delclass) = [];
end

for cidx = 1:length(classes)
    % relabel classes
    class_lbl = cidx-1;
    if length(classes{cidx}) > 1
        % combine classes
        y(ismember(y, classes{cidx})) = class_lbl;
    else
        y(y == classes{cidx}) = class_lbl;
    end
end

end

