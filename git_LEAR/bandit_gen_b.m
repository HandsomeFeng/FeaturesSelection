%% 8/2016 By Onur
% Selective select population of the data from supervised. 
% Run a logistic regression and get the coefficient
% use ratio_gen of whole dataset to generate logistic regression model
% the rest (1-ratio_gen) part can do the training, validation and testing
% INPUT
% X/Y: dataset.X(n*l_x),Y(n*1)
% ratio_gen: the rate of model-generation dataset
% OUTPUT
% B_0: Logistic Regression model coefficient
% acc_log: accuracy on the left dataset
% X_left/Y_left: the left dataset(size:(1-ratio_gen)*n)

function [B_0, acc_log, X_left, Y_left] = bandit_gen_b(X, Y, ratio_gen)
l_a = length(unique(Y));
indices = cell(1, l_a); 
size_a = zeros(1, l_a); 
indices_gen = [];
indices_left = [];
% for each label,get ratio_gen of them into generate set
for i = 1:l_a 
    indices{i} = find(Y == i); 
    size_a(i) = length(indices{i}); 
    perm_a = randperm(size_a(i)); 
    indices{i} =indices{i}(perm_a);
    indices_gen = [indices_gen indices{i}(1:floor(size_a(i)*ratio_gen))'];
    indices_left = [indices_left indices{i}(floor(size_a(i)*ratio_gen)+1:end)'];
end
X_gen = X(indices_gen, :); 
Y_gen = Y(indices_gen, :); 
X_left = X(indices_left, :); 
Y_left = Y(indices_left, :); 
n_left = length(Y_left); 
% generate the logistic regression model
B_0 = mnrfit(X_gen,Y_gen); 
% calculate the accuracy of logistic regresson on left dataset
p = exp([ones(n_left,1) X_left]*B_0);
p_last = 1 ./ (sum(p,2)+1);
pp = [p .* repmat(p_last,1,l_a-1), p_last];
[~,y_p] = max(pp, [], 2);
acc_logistic = sum(Y_left == y_p);

acc_log = acc_logistic/n_left; 
end 