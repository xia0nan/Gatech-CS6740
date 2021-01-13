function [err_train, err_test] = ModelKnn(train, test, K)

% load('usps-2cls.mat')
% [train, test] = SplitData(.8);
% K:= K-nn hyperparameter

[~, d] = size(train);

xtrain = double(train(:, 1: d-1));
ytrain = double(train(:, d));

xtest = double(test(:, 1: d-1));
ytest = double(test(:, d));

%%
mdl = fitcknn(xtrain,ytrain,'NumNeighbors',K,'Standardize',1);
ypredict_train = predict(mdl, xtrain);
ypredict_test = predict(mdl, xtest);

err_train = sum(ytrain~=ypredict_train)/length(ytrain);
err_test = sum(ytest~=ypredict_test)/length(ytest);

end