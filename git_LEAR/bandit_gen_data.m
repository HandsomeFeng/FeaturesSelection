%% 8/2016 By Onur
% generate logged bandit dataset
% use kappa * B_0 in a softmax
% kappa can help to overcome imbalance
% INPUT
% X/Y: dataset.X(n*l_x),Y(n*1)
% B_0: Logistic Regression model coefficient
% kappa: coefficient that times B_0 to generate bandit dataset
% OUTPUT
% X_b/A_b/D_b/P_b/Y_b: logged bandit dataset

function [X_b, Y_b, A_b, D_b, P_b] = bandit_gen_data(X, Y, B_0, kappa)
[n,~] = size(X);
l_a = length(unique(Y));
B_0 = kappa * B_0; 
X_b = X;
Y_b = Y;
A_b = zeros(n,l_a);         % A_b is a bit-vector,with 1(1) and 0(l_a-1)
D_b = zeros(n,1);
P_b = zeros(n,1); 
p = exp([ones(n,1) X]*B_0);
p_last = 1 ./ (sum(p,2)+1);
pp = [p .* repmat(p_last,1,l_a-1), p_last];
judge_p = sum(isnan(pp),2)==0;
tempA = zeros(n,1);
for i = 1:n
    if judge_p(i)
        tempA(i) = randsrc(1,1,[1:l_a ; pp(i,:)]);
        P_b(i) = pp(i,tempA(i));
        A_b(i,tempA(i)) = 1;
    else
        tempA(i) = randsrc(1,1,(1/l_a)*ones(1,l_a));
        P_b(i) = 1/l_a;
        A_b(i,tempA(i)) = 1;
    end
end
D_b(find(tempA == Y)) = -1;
end