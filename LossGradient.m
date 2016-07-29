%% 7/25/2016 By Qiaojun

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
d = l_x*l_a;
% for multi-labels
% z_a = [];
% for i = 0:l_a
%     z_a = [z_a;unique(perms([ones(1,i) zeros(1,l_a-i)]),'rows')];
% end
% for specific(one)-label
z_a = unique(perms([ones(1,1) zeros(1,l_a-1)]),'rows');

u = zeros(n,1);
g = zeros(n,d);
for i = 1:n
    type_A = find(A(i,:)==1);
    temp = Ra{type_A};
    XX = zeros(1,l_x);
    XX(temp) = X(i,temp);
    Phi = kron(XX,A(i,:));
    Phii = kron(XX,z_a);
    z_e = w * Phii';
    z = sum(exp(z_e));
    u(i) = Delta(i) * min(exp(w*Phi')/(P(i)*z),M);
    g(i,:) = Delta(i) / P(i) * u(i) * (Phi - sum(Phii.*repmat((exp(w*Phii')/z)',1,d)));
end