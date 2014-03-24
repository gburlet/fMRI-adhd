function [hmm] = hmmtrain_gmm(Xtrain, N, K, random_init)
%HMMTRAIN_GMM trains an HMM with a Gaussian mixture model emission distribution
%   from the training data Xtrain with the number of states N and number of mixtures K. 
%   random_init is a boolean specifying whether to randomly initialize certain model parameters. 

    % preprocess the observations to be used by the EM learning function
    Obs = permute(Xtrain, [1,3,2]);
    Obs = cellfun(@squeeze, num2cell(Obs, [2,3]), 'UniformOutput', false);

    vars = {};      % by default, hmmFit function randomizes initial model parameters
    if ~random_init
        % equiprobable state distributions
        pi0 = ones(N,1)/N;
        A0 = ones(N)/N;

        vars = {'pi0', pi0, 'trans0', A0};
    end

    % train HMM
    hmm = hmmFit(Obs, N, 'mixGaussTied', 'nmix', K, ...
                 'maxIter', 1000, 'convTol', 1e-7, ...
                 'nRandomRestarts', 3, 'verbose', true, vars{:});
end
