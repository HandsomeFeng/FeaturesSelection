%% 7/24/2016 By Qiaojun
% 1.LossGradient(Ds,w)
% 2.ABC(D,w,lamda)
% 3.SGD

function w = POEM(X,A,Delta,P,lamda,M,r,Ra)
tic
[n,l_x] = size(X);
[~,l_a] = size(A);
d = l_x * l_a;
w = zeros(1,d);
h = ones(1,d);
old_X = X;
old_A = A;
old_Delta = Delta;
old_P = P;
G = 10;                 %num of group
l_g = floor(n/10);
flag = 0;
r_time = 0;
while 1
    [A1,B1,~] = ABC(X,A,Delta,P,w,lamda,M,Ra);
%     rng default;
    temp = randperm(n);
    % shuffle D
    new_X(1:n,:) = old_X(temp,:);
    new_A(1:n,:) = old_A(temp,:);
    new_Delta(1:n,:) = old_Delta(temp,:);
    new_P(1:n,:) = old_P(temp,:);
    temps = 0;
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
        temps = temp + temps;
        if temp < 1e-4
            toc
            flag = 1;
            break;
        end
        w = w + r*nabla;
    end
    if flag == 1
        break;
    else
        temp = temps/G;
        r_time = r_time + 1;
        if r_time > 200
            toc
            break;
        end
    end
end