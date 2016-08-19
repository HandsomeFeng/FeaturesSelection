%% 8/2016 By Onur and Qiaojun
% Train and validate code
% X: feature matrix
% A: action vector
% R: reward
% Y: label
% r1_set, r2_set, r3_set : parameters
% type : 1 - with LEAR then res (l_r1, l_r2, l_r3) matrix, 0-without LEAR
% res (l_r3) matrix

function [acc_log,acc_LEAR,acc_LEAR_SN,acc_noLEAR,ra,ra_SN,Ra,Ra_SN] = ...
    LEAR_analysis(X, Y,m, r1_set,r2_set,r3_set,type,learning,...
    ratio,kappa,crossnum,test_iter)
ra = zeros(crossnum,2);
ra_SN = zeros(crossnum,2);
Ra = cell(crossnum,1);
Ra_SN = cell(crossnum,1);

acc_log = 0;
% 1.generate Logistic Regression model and test its accuracy
while acc_log < 0.7
    [B_0, acc_log, X_left, Y_left] = bandit_gen_b(X, Y, ratio);
end
% 2.generate logged bandit dataset
[X_b,Y_b,A_b,D_b,P_b] = bandit_gen_data(X_left, Y_left, B_0, kappa);

l_a= length(unique(Y_left));
[n,l_x] = size(X_left);
% seperate the data into different set
indices = crossvalind('Kfold',n,crossnum);
acc_LEAR = 0;
acc_LEAR_SN = 0;
acc_noLEAR = 0;
Ra = cell(crossnum,1);
Ra_SN = cell(crossnum,1);
RR = cell(1,l_a);                           % RR is for the non-LEAR test
for ii = 1:l_a
    RR{ii} = 1:l_x;
end
% for the n-cross validation, do n times iteration.
% Loop k: use set k as test-set and set k+1 validation-set. The other (n-2)
% set are train-set. After this, k = k + 1. Do it n times.
for index = 1:crossnum
    test_indices = find(indices == index);
    validate_indices = find(indices == mod(index,crossnum)+1);
    train_indices = setdiff(1:n, test_indices);
    train_indices = setdiff(train_indices, validate_indices);
    X_train= X_b(train_indices, :);
    A_train= A_b(train_indices, :);
    D_train= D_b(train_indices);
    P_train= P_b(train_indices);
    X_validate = X_b(validate_indices, :);
    Y_validate = Y_b(validate_indices);
    X_test = X_b(test_indices, :);
    Y_test = Y_b(test_indices);   
    % using the training set and validation set to get the Ra & w
    [~,ra(index,:),Ra{index},w_LEAR] = train(X_train,A_train,D_train,P_train,...
    X_validate,Y_validate,r1_set,r2_set,r3_set,m,type,0,learning);    
    [~,ra_SN(index,:),Ra_SN{index},w_LEAR_SN] = train(X_train,A_train,D_train,P_train,...
    X_validate,Y_validate,r1_set,r2_set,r3_set,m,type,1,learning);
    w_noLEAR = POEM(X_train,A_train,-D_train,P_train,r3_set,m,learning,RR);
    % test the result on the testing dataset
    acc_LEAR = acc_LEAR + test(X_test,Y_test,Ra{index},w_LEAR,l_a);
    acc_LEAR_SN = acc_LEAR_SN + test(X_test,Y_test,Ra_SN{index},w_LEAR_SN,l_a);
    acc_noLEAR = acc_noLEAR + test(X_test,Y_test,RR,w_noLEAR,l_a);
end
% get the average
acc_LEAR = acc_LEAR/crossnum;
acc_LEAR_SN = acc_LEAR_SN/crossnum;
acc_noLEAR = acc_noLEAR/crossnum;
end