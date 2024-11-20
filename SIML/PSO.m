
%% ��������Ⱥ�Ż��㷨PSO��װ����
function [gbestScore,gbest,fitnessbest]=PSO(n,K,x_lb,x_ub,narvs,Obj_fun)

%% ����Ⱥ�㷨�е�Ԥ����������������ò��ǹ̶��ģ������ʵ��޸ģ�
c1 = 2;  % ÿ�����ӵĸ���ѧϰ���ӣ�Ҳ��Ϊ������ٳ���
c2 = 2;  % ÿ�����ӵ����ѧϰ���ӣ�Ҳ��Ϊ�����ٳ���
w = 0.9;  % ����Ȩ��
vmax = ones(1,narvs).*(x_ub-x_lb).*0.15; % ���ӵ�����ٶ�


%% ��ʼ�����ӵ�λ�ú��ٶ�
% �����ʼ���������ڵ�λ���ڶ�������
x = repmat(x_lb, n, 1) + repmat((x_ub-x_lb), n, 1).*rand(n,narvs);
% �����ʼ�����ӵ��ٶȣ�������������Ϊ[-vmax,vmax]��
v = -vmax + 2*vmax .* rand(n,narvs);


%% ������Ӧ��(ע�⣬��Ϊ����С�����⣬������Ӧ��ԽСԽ��)
fit = zeros(n,1);  % ��ʼ����n�����ӵ���Ӧ��ȫΪ0
for i = 1:n  % ѭ����������Ⱥ������ÿһ�����ӵ���Ӧ��
    fit(i) = Obj_fun(x(i,:));   % ����Obj_fun������������Ӧ��
end
pbest = x;   % ��ʼ����n����������Ϊֹ�ҵ������λ�ã���һ��n*narvs��������
ind = find(fit == min(fit), 1);  % �ҵ���Ӧ����С���Ǹ����ӵ��±�
gbest = x(ind,:);  % ����������������Ϊֹ�ҵ������λ�ã���һ��1*narvs��������


%% ����K���������ٶ���λ��
fitnessbest = ones(K,1);  % ��ʼ��ÿ�ε����õ�����ѵ���Ӧ��
for t = 1:K  % ��ʼ������һ������K��
    for i = 1:n   % ���θ��µ�i�����ӵ��ٶ���λ��
        v(i,:) = w*v(i,:) + c1*rand(1)*(pbest(i,:) - x(i,:)) + c2*rand(1)*(gbest - x(i,:));  % ���µ�i�����ӵ��ٶ�
        % ������ӵ��ٶȳ���������ٶ����ƣ��Ͷ�����е���
        for j = 1: narvs
            if v(i,j) < -vmax(j)
                v(i,j) = -vmax(j);
            elseif v(i,j) > vmax(j)
                v(i,j) = vmax(j);
            end
        end
        x(i,:) = x(i,:) + v(i,:); % ���µ�i�����ӵ�λ��
        % ������ӵ�λ�ó����˶����򣬾Ͷ�����е���
        for j = 1: narvs
            if x(i,j) < x_lb(j)
                x(i,j) = x_lb(j);
            elseif x(i,j) > x_ub(j)
                x(i,j) = x_ub(j);
            end
        end
        fit(i) = Obj_fun(x(i,:));  % ���¼����i�����ӵ���Ӧ��
        if fit(i) < Obj_fun(pbest(i,:))   % �����i�����ӵ���Ӧ��С�������������Ϊֹ�ҵ������λ�ö�Ӧ����Ӧ��
            pbest(i,:) = x(i,:);   % �Ǿ͸��µ�i����������Ϊֹ�ҵ������λ��
        end
        if  fit(i) < Obj_fun(gbest)  % �����i�����ӵ���Ӧ��С�����е���������Ϊֹ�ҵ������λ�ö�Ӧ����Ӧ��
            gbest = pbest(i,:);   % �Ǿ͸���������������Ϊֹ�ҵ������λ��
        end
    end
    fitnessbest(t) = Obj_fun(gbest);  % ���µ�d�ε����õ�����ѵ���Ӧ��
    if mod(t,5)==0
        disp(['PSO��' num2str(t) '�ε���'])
    end
end
gbestScore=fitnessbest(K);
end