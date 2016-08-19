%% 7/24/2016 By Qiaojun
% 1.LossGradient(Ds,w)
% 2.ABC(D,w,lamda)
% 3.SGD

function w = POEM(X,A,Delta,P,lamda,M,r,Ra)
% tic
[n,~] = size(X);
[~,l_a] = size(A);
w =[];
for i = 1:l_a
    w = [w zeros(1, length(Ra{i})+1)];
end
d = length(w);
h = ones(1,d);
old_X = X;
old_A = A;
old_Delta = Delta;
old_P = P;
% You can change the G here to use part of the data to calculate gradient
% at a time
G = 3;                      %num of group(now we use 1)
l_g = floor(n/G);
flag = 0;
u_old = 0;
r_time = 0;
while 1
    [A1,B1,~] = ABC(X,A,Delta,P,w,lamda,M,Ra);
    %     rng default;
    temp = randperm(n);
    new_X = old_X(temp,:);
    new_A = old_A(temp,:);
    new_Delta = old_Delta(temp);
    new_P = old_P(temp);
    for i = 1:G
        if i ~= G
            [u,g] = LossGradient(new_X((i-1)*l_g+1:i*l_g,:),new_A((i-1)*l_g+1:i*l_g,:),...
                new_Delta((i-1)*l_g+1:i*l_g,:),new_P((i-1)*l_g+1:i*l_g,:),w,M,Ra);
        else
            [u,g] = LossGradient(new_X((i-1)*l_g+1:n,:),new_A((i-1)*l_g+1:n,:),...
                new_Delta((i-1)*l_g+1:n,:),new_P((i-1)*l_g+1:n,:),w,M,Ra);
        end
        u = mean(u);
        g = mean(g);
        h = g.^2 + h;
        j = g ./ sqrt(h);
        nabla = A1.*j + 2*B1*u*j;
        temp = norm(nabla);       
        if temp < 2*1e-2 && abs(u_old - u) < 1*1e-3...
                || r_time > 300              
            flag = 1;
            break;
        end
        u_old= u; 
        w = w + r*nabla;
    end
    if flag == 1
        break; 
    end
    r_time = r_time + 1;
end
end