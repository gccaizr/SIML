
%% 基础粒子群优化算法PSO封装函数
function [gbestScore,gbest,fitnessbest]=PSO(n,K,x_lb,x_ub,narvs,Obj_fun)

%% 粒子群算法中的预设参数（参数的设置不是固定的，可以适当修改）
c1 = 2;  % 每个粒子的个体学习因子，也称为个体加速常数
c2 = 2;  % 每个粒子的社会学习因子，也称为社会加速常数
w = 0.9;  % 惯性权重
vmax = ones(1,narvs).*(x_ub-x_lb).*0.15; % 粒子的最大速度


%% 初始化粒子的位置和速度
% 随机初始化粒子所在的位置在定义域内
x = repmat(x_lb, n, 1) + repmat((x_ub-x_lb), n, 1).*rand(n,narvs);
% 随机初始化粒子的速度（这里我们设置为[-vmax,vmax]）
v = -vmax + 2*vmax .* rand(n,narvs);


%% 计算适应度(注意，因为是最小化问题，所以适应度越小越好)
fit = zeros(n,1);  % 初始化这n个粒子的适应度全为0
for i = 1:n  % 循环整个粒子群，计算每一个粒子的适应度
    fit(i) = Obj_fun(x(i,:));   % 调用Obj_fun函数来计算适应度
end
pbest = x;   % 初始化这n个粒子迄今为止找到的最佳位置（是一个n*narvs的向量）
ind = find(fit == min(fit), 1);  % 找到适应度最小的那个粒子的下标
gbest = x(ind,:);  % 定义所有粒子迄今为止找到的最佳位置（是一个1*narvs的向量）


%% 迭代K次来更新速度与位置
fitnessbest = ones(K,1);  % 初始化每次迭代得到的最佳的适应度
for t = 1:K  % 开始迭代，一共迭代K次
    for i = 1:n   % 依次更新第i个粒子的速度与位置
        v(i,:) = w*v(i,:) + c1*rand(1)*(pbest(i,:) - x(i,:)) + c2*rand(1)*(gbest - x(i,:));  % 更新第i个粒子的速度
        % 如果粒子的速度超过了最大速度限制，就对其进行调整
        for j = 1: narvs
            if v(i,j) < -vmax(j)
                v(i,j) = -vmax(j);
            elseif v(i,j) > vmax(j)
                v(i,j) = vmax(j);
            end
        end
        x(i,:) = x(i,:) + v(i,:); % 更新第i个粒子的位置
        % 如果粒子的位置超出了定义域，就对其进行调整
        for j = 1: narvs
            if x(i,j) < x_lb(j)
                x(i,j) = x_lb(j);
            elseif x(i,j) > x_ub(j)
                x(i,j) = x_ub(j);
            end
        end
        fit(i) = Obj_fun(x(i,:));  % 重新计算第i个粒子的适应度
        if fit(i) < Obj_fun(pbest(i,:))   % 如果第i个粒子的适应度小于这个粒子迄今为止找到的最佳位置对应的适应度
            pbest(i,:) = x(i,:);   % 那就更新第i个粒子迄今为止找到的最佳位置
        end
        if  fit(i) < Obj_fun(gbest)  % 如果第i个粒子的适应度小于所有的粒子迄今为止找到的最佳位置对应的适应度
            gbest = pbest(i,:);   % 那就更新所有粒子迄今为止找到的最佳位置
        end
    end
    fitnessbest(t) = Obj_fun(gbest);  % 更新第d次迭代得到的最佳的适应度
    if mod(t,5)==0
        disp(['PSO第' num2str(t) '次迭代'])
    end
end
gbestScore=fitnessbest(K);
end