%% 8/2016 By Onur and Qiaojun
% calculate the predict accuracy using softmax
% INPUT:
% X/Y:      data 
% R_a:      relevant features
% w:        corresponding coefficient
% l_a:      the number of action
% OUTPUT:
% accuracy: accuracy result

function accuracy = test(X,Y,R_a,w,l_a)
[n,l_x] = size(X);
w_mask = zeros(l_x+1,l_a);
w_begin = 1;
for i = 1:l_a
    w_temp = length(R_a{i});
    w_mask(R_a{i}+1,i) = w(w_begin+1:w_begin+w_temp);
    w_mask(1,i) = w(w_begin);
    w_begin = w_begin+w_temp+1;
end
hh = [];
p = [ones(n,1) X] * w_mask;
p = exp(p)./repmat(sum(exp(p),2),1,l_a);
[~,Y_p] = max(p,[],2);
ctimes = length(find(Y_p==Y));
hh = [hh;Y_p(find(Y_p==Y))];
accuracy = ctimes/n;
% hh can show how many of each label are predicted correctly
% hist(hh);
end