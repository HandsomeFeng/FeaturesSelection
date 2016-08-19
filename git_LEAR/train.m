%% 8/2016 By Onur and Qiaojun
% INPUT
% X_train,A_train,D_train,P_train:      training dataset
% X_validate,Y_validate:                validation dataset
% r1_set,r2_set,r3_set:                 r1&r2-LEAR,r3-POEM
% m,type:                               parameter in LEAR
% SN:                                   SN=0 normal LEAR/SN=1 LEAR_SN

function [accuracy,ra,Ra,w] = train(X_train,A_train,D_train,P_train,...
    X_validate,Y_validate,r1_set,r2_set,r3_set,m,type,SN,learning)
l_r1 = length(r1_set);
l_r2 = length(r2_set);
l_r3 = length(r3_set);
l_x = size(X_train,2);
l_a = size(A_train,2);
rr = [];
R_anum = 0;
Racode = [];
R_a = {};
w = {};
avg_acc = [];
for r1 = 1:l_r1
    for r2 = 1:l_r2 
        if SN ~= 1
            R_hat = LEAR(X_train,A_train,D_train,P_train,r1_set(r1),r2_set(r2),m,type);
        else
            R_hat = LEAR_SN(X_train,A_train,D_train,P_train,r1_set(r1),r2_set(r2),type);
        end
        tempcode = zeros(l_a,l_x);
        for i = 1:l_a
            tempindex = R_hat{i};
            tempcode(i,tempindex) = 1;
        end
        tempcode = reshape(tempcode',[1,l_a*l_x]);
        flag = 0;
        for i = 1:size(Racode,1)
            if tempcode == Racode(i,:)
                flag = 1;
                break;
            end
        end
        if flag == 0
            R_anum = R_anum+1;
            Racode = [Racode;tempcode];
            rtemp = [r1_set(r1),r2_set(r2)];
            rr = [rr;rtemp];
            R_a{R_anum} = R_hat;
            w{R_anum} = POEM(X_train,A_train,-D_train,P_train,r3_set,m,learning,R_hat);
            avg_acc(R_anum) = test(X_validate,Y_validate,R_hat,w{R_anum},l_a);
        end
    end
end
% find the best Ra&w on the validation dataset
accuracy = max(avg_acc);
candidate = find(avg_acc==accuracy);
if length(candidate) > 1
    temp = sum(Racode(candidate,:),2);
    IND = find(temp == min(temp));
    if length(IND) > 1
        [~,INDD] = max(rr(candidate(IND),1));
        IND = candidate(IND(INDD));
    else
        IND = candidate(IND);
    end
else
    IND = candidate;
end
Ra = R_a{IND};
ra = rr(IND,:);
w = w{IND};
end