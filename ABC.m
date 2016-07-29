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

function [A,B,C] = ABC(X,A,Delta,P,w,lamda,M,Ra)
[n,~] = size(X);
[u,~] = LossGradient(X,A,Delta,P,w,M,Ra);
R = mean(u);
V = sqrt(sum((u-R).^2./(n-1)));
A = 1 - lamda*sqrt(n)*R/((n-1)*V);
B = lamda/(2*(n-1)*V*sqrt(n));
C = lamda*V/(2*sqrt(n)) + lamda*sqrt(n)*R^2/(2*(n-1)*V);