%% 7/28/2016 By Qiaojun
% shuffle the index of train/valid/test
n1 = size(index_train,1);
n2 = size(index_valid,1);
n3 = size(index_test,1);
temp = [index_train;index_valid;index_test];
temp1 = randperm(n1+n2+n3,n1);
index_train = temp(temp1);
temp(temp1) = [];
temp2 = randperm(n2+n3,n2);
index_valid = temp(temp2);
temp(temp2) = [];
index_test = temp;