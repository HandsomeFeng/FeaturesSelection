%% 8/2016 By Onur and Qiaojun
% This program show the performance of LEAR/LEAR_SN/no-LEAR
clear;
% these are the dataset we use
dataset = {'data/vehicle.mat','data/pendigits_tes.mat',...
    'data/optdigits_tra_32.mat','data/satimage_tra.mat'};
size_dataset = size(dataset,2);
% these are the coefficient of LEAR or POEM. r1&r2-LEAR,r3-POEM.r3 can be 1
% value
r1_set = 0.0:0.2:3.0; 
r2_set = 0:0.02:0.10; 
r3_set = 10.^(1:1:1);
% some other coefficient:
% m: the bound in LEAR
% kappa: the coefficient used to generate bandit dataset
% ratio_gen: decide how mand data are used for generate model
% learning: the learning ratio(step) in POEM
% ITER: do ITER times iteration for one dataset(each time different bandit
% dataset)
% crossnum: decide n-cross validation in the LEAR_analysis
m = 100; 
kappa = 0.2; 
ratio_gen = 0.4; 
learning = 5;
ITER = 5;
crossnum = 3;
% record the result
result = zeros(size_dataset,4);
ra = cell(2,ITER,size_dataset);
Ra = cell(2,ITER,size_dataset);

% for i = 1:size_dataset
for i = 1:1
    load(dataset{i});
    [n,l_x] = size(X);
    % normalized dataset can get better result
    X = (X-repmat(min(X),n,1))./repmat(max(X)-min(X),n,1);    
    % judge whether continious or discrete by counting how many different
    % values
    type = [];
    for j = 1:l_x
        if length(unique(X(:,j)))<5
            type = [type,'d'];
        else
            type = [type,'c'];
        end
    end
    acc_log = zeros(1, ITER);
    acc_LEAR = zeros(1, ITER);
    acc_LEAR_SN = zeros(1, ITER);
    acc_noLEAR = zeros(1, ITER); 
    % do the iteration and the bandid dataset is different each time
    for j = 1:ITER
        tic
        [acc_log(j),acc_LEAR(j),acc_LEAR_SN(j),acc_noLEAR(j),...
            ra{1,j,i},ra{2,j,i},Ra{1,j,i},Ra{2,j,i}] = LEAR_analysis...
            (X, Y,m, r1_set,r2_set,r3_set, type, learning,ratio_gen,kappa,crossnum);    
        toc
    end
    % get the average
    result(i,1) = sum(acc_log)/ITER;                % logistic regression
    result(i,2) = sum(acc_LEAR)/ITER;               % LEAR
    result(i,3) = sum(acc_LEAR_SN)/ITER;            % LEAR_SN
    result(i,4) = sum(acc_noLEAR)/ITER;             % Only POEM
end
size_vehicle = [18 4];
acc_vehicle = result(1,:);
Ra_vehicle = Ra(:,:,1);