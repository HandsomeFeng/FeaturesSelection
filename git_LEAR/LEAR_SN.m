%% 7/22/2016 By Qiaojun
% LEAR with Self-Normalizing Estimators
% INPUT
% X:d-dimensional feature space(n*d)
% A:k-option action set(n*1)
% R:result set(n*1)
% P:n*1
% r1,r2 are parameters to add restrictions
% type: 'd'(discrete) OR 'c'(continious). When it is continious
% OUTPUT
% Rhat: l_a*1 cell. In each cell there are the index of features that are
% related to this action.

function Rhat = LEAR_SN(X,A,R,P,r1,r2,type)

%step 1:preprocessing
[n,d] = size(X);
nn = floor(n^(1/3));
 
for i = 1:d
    if type(i) == 'c'
        maxi = max(X(:,i));
        mini = min(X(:,i));
        if maxi ~= mini
            X(:,i) = ceil ((X(:,i) - mini.*ones(n,1)) ./ (maxi-mini) .* nn);
        else
            X(:,i) = ones(n,1);
        end
    end
end

value_d = cell(1,d);                        % value_d records the values in every features
l_d = zeros(1,d);                           % l_d records the number of sorts in d features
n_d = cell(1,d);                            % n_d records number of people with specific feature
if size(A,2) > 1                            % change from bitvector to integer
    l_a = size(A,2);
    AA = zeros(size(A,1),1);
    for i = 1:l_a
        temp = find(A(:,i)==1);
        AA(temp) = i;
    end
    A = AA;
    clear AA;
    value_a = unique(A);
else
    value_a = unique(A);                    % value_d records the values of actions
    l_a = length(value_a);                  % l_a records num of actions
end
for i = 1:d
    value_d{i} = unique(X(:,i));
    l_d(i) = length(unique(X(:,i)));    
    n_d{i} = zeros(l_d(i),1);
    for j = 1:l_d(i)
        n_d{i}(j) = length(find(X(:,i) == value_d{i}(j)));
    end
end

% step 2:estimating rewards(based on action)-get u and u_a(R and R_a)
% validation needed
u = zeros(l_a,1);
u_x = cell(l_a,d);
pj = cell(l_a,d);
% i-index of feature;j-index of action;k-index of patient
for i = 1:l_a
    xx = find(A(:)==value_a(i));
	u(i,1) = sum(R(xx)./max(P(xx),0.01)) / sum(1./P(xx));   
    for j = 1:d
        u_x{i,j} = zeros(l_d(1,j),1);
        pj{i,j} = zeros(l_d(1,j),1);        
        for k = 1:l_d(j)
            l = find(X(:,j)==value_d{j}(k));
            ll = ismember(l,xx);
            u_x{i,j}(k) = sum(R(l(ll)) ./ P(l(ll)));
            pj{i,j}(k) = sum(1 ./ P(l(ll)));
        end
        u_x{i,j}(:,1) = u_x{i,j}(:,1) ./ pj{i,j}(:);
        u_x{i,j}(find(isnan(u_x{i,j}))) = 0;
    end
end

% step 3:get Gm
Gm = zeros(l_a,d);
for i = 1:l_a
    for j = 1:d
        for k = 1:l_d(1,j)
            Gm(i,j) = abs(u_x{i,j}(k) - u(i,1)) * n_d{j}(k) / n + Gm(i,j);
        end
    end
end

% step 4:get V
V = zeros(l_a,d);
Vn = cell(l_a,d);
for i = 1:l_a
    for j = 1:d
        Vn{i,j} = zeros(l_d(j),1);
        pv = zeros(l_d(j),1);
        for k = 1:l_d(j)
            l = find(A(:)==value_a(i));
            Vn{i,j}(k) = sum((R(l)-u_x{i,j}(k)).^2.*(1./P(l)).^2);
            pv(k) = sum((1./P(l)).^2);
        end
        Vn{i,j}(:) = Vn{i,j}(:) ./ pv(:);
        V(i,j) = sum(n_d{j}(:) .* Vn{i,j}(:) ./ n);
    end 
end

% step 5:get the result of Rhat
Rhat = cell(l_a,1);
for i = 1:l_a
    for j = 1:d
        k = 1:d;
        k(j) = [];
        temp = abs(corr(X(:,j),X(:,k)));
        temp(isnan(temp)) = 0;
        Cor = sum(temp);
        if type(j) == 'c'
            tempR = Gm(i,j) - r1 * (sqrt(V(i,j)) / floor(n^(1/3))) - r2 / (d - 1) * Cor;
        else
            tempR = Gm(i,j) - r1 * sqrt(l_d(1,j) * V(i,j) / n) - r2 / (d - 1) * Cor;
        end
        if tempR >= 0
            Rhat{i} = [Rhat{i},j];
        end
    end
end