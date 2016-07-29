%% 7/28/2016 By Qiaojun
% use a multivariate logistic regression model to generate bandit data
n1 = size(index_train,1);
n2 = size(index_valid,1);
n3 = size(index_test,1);
n = n1 + n2 + n3;
XX = [X(index_train,:);X(index_valid,:);X(index_test,:)];
YY = [Y(index_train,:);Y(index_valid,:);Y(index_test,:)];
l_a = length(unique(YY));
A = zeros(n,l_a);                           % A should be a bitvector(n*l_a)
D = zeros(n,1);
P = zeros(n,1);
tempx = [ones(n,1) XX];
p = tempx * B;                              % B was got from multivariate logistic regression
p = exp(p);
temp = sum(p,2) + 1;
p = [p./repmat(temp,1,l_a-1) 1./temp];
for i = 1:n
    temp = randsrc(1,1,[1:l_a ; p(i,:)]);
    A(i,temp) = 1;
    P(i) = p(i,temp);
    if temp == YY(i)
        D(i) = -1;                          % action == label,D=-1.else D=0
    end
end
accu_logistic = sum(D)/(-1)/n;              % accuracy of logistic regression
X_train = XX(1:n1,:);
A_train = A(1:n1,:);
D_train = D(1:n1,:);
P_train = P(1:n1,:);
X_valid = XX(n1+1:n1+n2,:);
A_valid = A(n1+1:n1+n2,:);
D_valid = D(n1+1:n1+n2,:);
P_valid = P(n1+1:n1+n2,:);
X_test = XX(n1+n2+1:n1+n2+n3,:);
A_test = A(n1+n2+1:n1+n2+n3,:);
D_test = D(n1+n2+1:n1+n2+n3,:);
P_test = P(n1+n2+1:n1+n2+n3,:);
Y_valid = Y(index_valid,:);
Y_test = Y(index_test,:);
clear A B D P XX YY p temp tempx n1 n2 n3;