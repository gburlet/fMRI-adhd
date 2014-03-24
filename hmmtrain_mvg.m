function [hmm] = hmmtrain_mvg(Xtrain, N, random_init)
%HMMTRAIN_MVG trains an HMM with a multivariate gaussian emission distribution
%   from the training data Xtrain with the number of states N. 
%   random_init is a boolean specifying whether to approximate certain parameters 
%   from the training data.

    % preprocess the observations to be used by the EM learning function
    Obs = permute(Xtrain, [1,3,2]);
    Obs = cellfun(@squeeze, num2cell(Obs, [2,3]), 'UniformOutput', false);

    vars = {};      % by default, hmmFit function randomizes initial model parameters
    if ~random_init
        % derive initial model parameters from training data
        Xtrain_tempavg = squeeze(mean(Xtrain,2));
        mu0 = repmat(mean(Xtrain_tempavg, 1), [N,1])';
        Sigma0 = repmat(cov(Xtrain_tempavg), [1,1,N]);
        emission0 = condGaussCpdCreate(mu0, Sigma0); 

        % equiprobable state distributions
        pi0 = ones(N,1)/N;
        A0 = ones(N)/N;

        vars = {'pi0', pi0, 'trans0', A0, 'emission0', emission0};
    end

    % train HMM
    hmm = hmmFit(Obs, N, 'gauss', ...
                 'maxIter', 1000, 'convTol', 1e-7, ...
                 'nRandomRestarts', 3, 'verbose', true, vars{:});
end
