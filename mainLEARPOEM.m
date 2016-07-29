%% 7/28/2016 By Qiaojun
% combine the LEAR and POEM together

clear;
load('ecoli');                      % the raw data(decrease the num of label to 5)
                                    % X:features(n*l_x) Y:labels(n*1,1~5)
load('index');                      % the division of each group and 
                                    % logistic regression model's parameters
shuffle;                            % shuffle the index of train/valid/test
super2bandit;                       % generate bandit data

% step1:train
% these define how the lamda1 and lamda2 vary
% lamda1: start from b1, add s1 every time, add t1 times in total(together t1+1 values)
% lamda2: start from b2, add s2 every time, add t2 times in total(together t2+1 values)
b1 = 0.2;
s1 = 0.1;
t1 = 15;
b2 = 0.02;
s2 = 0.01;
t2 = 15;

l_x = size(X_train,2);
l_a = size(A_train,2);
SN = 1;                             % SN = 1:using Self Normalizing Estimators
                                    % SN = 0:not using SN
M = 100;                            % cycle for M times to test accuracy
features = cell((t1+1)*(t2+1),l_a+2);   % record what features to choose as R(a)
% lamda3 ranging from 1e(b3) to 1e(e3),that is now 0.0001 to 1(in multiples of 10)
b3 = -4;
e3 = 2;
l_try = (e3-b3+1)*(t1+1)*(t2+1);
w = zeros(l_try,l_x*l_a);           % record the w for LEAR(less features used)
w1 = zeros(e3-b3+1,l_x*l_a);        % record the w1 for non-LEAR(the same features)
coef = zeros(l_try,3);              % record the parameters for LEAR
coef1 = zeros(e3-b3+1,1);           % record the parameters for non-LEAR
num_w = 1;
num_w1 = 1;
RR = cell(1,l_a);                   % RR is for the non-LEAR step
for i = 1:l_a
    RR{i} = 1:l_x;
end

% try different lamda1,lamda2,lamda3
for ll3 = b3:e3
    l3 = 10^ll3;
    for l1 = b1:s1:b1+s1*t1
        for l2 = b2:s2:b2+s2*t2
            if SN
                Ra = LEAR_SN(X_train,A_train,-D_train,P_train,l1,l2,'c');
            else
                Ra = LEAR(X_train,A_train,-D_train,P_train,l1,l2,M,'c');
            end
            temp = round((l1-b1)/s1*(t1+1)+(l2-b2)/s2+1);
            for i = 1:l_a
                features{temp,i} = Ra{i};       % record R(a)
            end
            feature{temp,l_a+1} = l1;
            feature{temp,l_a+2} = l2;
            % using R(a) to reduce creatures
            w(num_w,:) = POEM(X_train,A_train,D_train,P_train,l3,M,1,Ra);
            coef(num_w,:) = [l1,l2,l3];
            num_w = num_w + 1;
        end
    end
    % here is using all origin featues
    w1(num_w1,:) = POEM(X_train,A_train,D_train,P_train,l3,M,1,RR);
    coef1(num_w1) = l3;
    num_w1 = num_w1 + 1;
end

% step2:validation
accu_LEAR = zeros(l_try,1);
accu_POEM = zeros(e3-b3+1,1);
L_TEST = 100;
z_a = eye(l_a);
n = size(X_valid,1);
for w_index = 1:l_try
	correct = 0;
    for cir_index = 1:L_TEST
        correct_num = 0;
        for i = 1:n
            tl1 = coef(w_index,1);
            tl2 = coef(w_index,2);
            temp = round((tl1-b1)/s1*(t1+1)+(tl2-b2)/s2+1);
            LX = features{temp,find(A_valid(i,:)==1)};
            XX = zeros(1,l_x);
            XX(LX) = X_valid(i,LX);
            Phii = kron(XX,z_a);           
            p = w(w_index,:) * Phii';
            p = exp(p)/sum(exp(p));
            temp = randsrc(1,1,[1:l_a ; p]);
            if Y_valid(i) == temp               % testing the result compared to real Y
                correct_num = correct_num + 1;
            end
        end
        correct = correct + correct_num/n;
    end
    accu_LEAR(w_index) = correct / L_TEST;
end

for w_index = 1:e3-b3+1
	correct = 0;
    for cir_index = 1:L_TEST
        correct_num = 0;
        for i = 1:n
            Phii = kron(X_valid(i,:),z_a);         
            p = w1(w_index,:) * Phii';
            p = exp(p)/sum(exp(p));
            temp = randsrc(1,1,[1:l_a ; p]);
            if Y_valid(i) == temp
                correct_num = correct_num + 1;
            end
        end
        correct = correct + correct_num/n;
    end
    accu_POEM(w_index) = correct / L_TEST;    
end

% 3.testing
n = size(X_test,1);
[~,index_max] = max(accu_LEAR);
w_max_LEAR = w(index_max,:);
correct = 0;
% first LEAR way
for cir_index = 1:L_TEST
    correct_num = 0;
    for i = 1:n
        tl1 = coef(index_max,1);
        tl2 = coef(index_max,2);
        temp = round((tl1-b1)/s1*(t1+1)+(tl2-b2)/s2+1);
        LX = features{temp,find(A_test(i,:)==1)};
        XX = zeros(1,l_x);
        XX(LX) = X_test(i,LX);
        Phii = kron(XX,z_a);
        p = w(w_index,:) * Phii';
        p = exp(p)/sum(exp(p));
        temp = randsrc(1,1,[1:l_a ; p]);
        if Y_test(i) == temp
            correct_num = correct_num + 1;
        end
    end
    correct = correct + correct_num/n;
end
accu_test_LEAR = correct / L_TEST;

% second,non-LEAR way
[~,index_max] = max(accu_POEM);
w_max_POEM = w1(index_max,:);
correct = 0;
for cir_index = 1:L_TEST
    correct_num = 0;
    for i = 1:n
        Phii = kron(X_test(i,:),z_a);
        p = w1(w_index,:) * Phii';
        p = exp(p)/sum(exp(p));
        temp = randsrc(1,1,[1:l_a ; p]);
        if Y_test(i) == temp
            correct_num = correct_num + 1;
        end
    end
    correct = correct + correct_num/n;
end
accu_test_POEM = correct / L_TEST;