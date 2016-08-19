%% 8/15/2016 By Qiaojun
% part of POEM
% INPUT
% Data:struct of 4 features
%   X:features
%   A:action
%   Delta:loss
%   P:possibility
% w:weight
% M:bound

% OUTPUT
% u:one number
% d:1*d vector

function [u,g] = LossGradient(X,A,Delta,P,w,M,Ra)
[n,l_x] = size(X);
[~,l_a] = size(A);
d = length(w);
u = zeros(n,1);
g = zeros(n,d);
% get a w_mask to calculate
w_mask = zeros(l_x+1,l_a);
x_mask = zeros(l_x,l_a);
j_begin = 1;
for j = 1:l_a
    j_temp = length(Ra{j});
	w_mask(1,j) = w(j_begin);
    w_mask(Ra{j}+1,j) = w(j_begin+1:j_begin+j_temp);
    x_mask(Ra{j},j) = 1;
    j_begin = j_begin + j_temp + 1;
end
x_mask = [ones(1,l_a);x_mask];

Phii = exp([ones(n,1) X] * w_mask);      	% get the possibility matrix
Phi = sum(Phii .* A,2);
temp = Phi./(P.*sum(Phii,2));
g0 = find(M <= temp);
u = Delta .* min(M,temp);
p_mask = A - Phii./ repmat(sum(Phii,2),1,l_a);
g_mask = [];
for i = 1:l_a;
    g_mask = [g_mask [ones(n,1) X].*(repmat(p_mask(:,i),1,l_x+1))];
end
g_mask = g_mask(:,logical(reshape(x_mask,[1,(l_x+1)*l_a])));
g = repmat(Delta ./ P .* u,1,size(g_mask,2)) .*(g_mask);
g(g0,:) = 0;
end