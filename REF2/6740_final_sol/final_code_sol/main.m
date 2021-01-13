% entry for 6740-19f-final-Q6

%Repeat the experiments for 100 times
tic
N = 100;

%Let p change from 0.1, 0.2, 0.5, 0.8, 0.9 to compare the performance of each classifier
pchoice =[0.1, 0.2, 0.5, 0.8, 0.9];

%% Bayes Classifer

for p = pchoice
    
    err_bayes = zeros(N,2);
%     p = 0.8;

    for i = 1 : N

        [train, test] = SplitData(p);

        [err_train, err_test] = ModelFull(train, test);
        err_bayes(i,:) = [err_train, err_test];
        
    end

    mean_err_bayes = mean(err_bayes);

    fprintf('err_bayes,   p = %g\n', p);
    fprintf('training_err: %g,   testing_err: %g\n', mean_err_bayes(1), mean_err_bayes(2));
end


%% Knn

K = [5, 10, 15, 30];    % choice of K

for k = K
    for p = pchoice

        err_knn = zeros(N,2);
    %     p = 0.8;

        for i = 1 : N

            [train, test] = SplitData(p);

            [err_train, err_test] = ModelKnn(train, test, k);
            err_knn(i,:) = [err_train, err_test];
        end

        mean_err_knn = mean(err_knn);

        fprintf('err_knn,   K = %g,  p = %g\n', k, p);
        fprintf('training_err: %g,   testing_err: %g\n', mean_err_knn(1), mean_err_knn(2));
    end
    
end
toc
%% . results
% err_knn,   K = 5,  p = 0.1
% training_err: 0.0392273,   testing_err: 0.0767424
% err_knn,   K = 5,  p = 0.2
% training_err: 0.0267727,   testing_err: 0.0524886
% err_knn,   K = 5,  p = 0.5
% training_err: 0.0160727,   testing_err: 0.0292182
% err_knn,   K = 5,  p = 0.8
% training_err: 0.011983,   testing_err: 0.0232273
% err_knn,   K = 5,  p = 0.9
% training_err: 0.0110253,   testing_err: 0.0210909
% err_knn,   K = 10,  p = 0.1
% training_err: 0.0727727,   testing_err: 0.102848
% err_knn,   K = 10,  p = 0.2
% training_err: 0.0498182,   testing_err: 0.0672273
% err_knn,   K = 10,  p = 0.5
% training_err: 0.0297545,   testing_err: 0.0399
% err_knn,   K = 10,  p = 0.8
% training_err: 0.0213125,   testing_err: 0.0271818
% err_knn,   K = 10,  p = 0.9
% training_err: 0.0218939,   testing_err: 0.0275
% err_knn,   K = 15,  p = 0.1
% training_err: 0.0747273,   testing_err: 0.0951818
% err_knn,   K = 15,  p = 0.2
% training_err: 0.0476818,   testing_err: 0.0640511
% err_knn,   K = 15,  p = 0.5
% training_err: 0.0307636,   testing_err: 0.0383
% err_knn,   K = 15,  p = 0.8
% training_err: 0.0235227,   testing_err: 0.0307045
% err_knn,   K = 15,  p = 0.9
% training_err: 0.0215303,   testing_err: 0.0275
% err_knn,   K = 30,  p = 0.1
% training_err: 0.122955,   testing_err: 0.138889
% err_knn,   K = 30,  p = 0.2
% training_err: 0.0794091,   testing_err: 0.093125
% err_knn,   K = 30,  p = 0.5
% training_err: 0.0446182,   testing_err: 0.0521636
% err_knn,   K = 30,  p = 0.8
% training_err: 0.0351648,   testing_err: 0.0386364
% err_knn,   K = 30,  p = 0.9
% training_err: 0.0325909,   testing_err: 0.0377727
