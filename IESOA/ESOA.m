

function [y_global_best, x_global_best, Convergence_curve]=ESOA(num1,num2,num3,SearchAgents_no, Max_iter, lb, ub, dim, fobj)


%% 运行群智能
func = fobj;

beta1 = 0.9;
beta2 = 0.99;

%Initialize the positions of search agents
lb=lb.*ones(1,dim);
ub=ub.*ones(1,dim);

%% 改进方式1：头部变异|混沌初始化
x=repmat(lb, SearchAgents_no, 1)+chaos(num1,SearchAgents_no,dim) .* repmat((ub-lb), SearchAgents_no, 1);

Convergence_curve=zeros(1,Max_iter);

w = random('Uniform', -1, 1, SearchAgents_no, dim);
%g = random('Uniform', -1, 1, SearchAgents_no, dim);
m = zeros(SearchAgents_no, dim);
v = zeros(SearchAgents_no, dim);
y = zeros(SearchAgents_no,1);
for i=1:SearchAgents_no
    y(i) = func(x(i,:));
end
p_y = y;

x_hist_best = x;
g_hist_best = x;
y_hist_best = ones(SearchAgents_no)*inf;

x_global_best = x(1, :);
g_global_best = zeros(1, dim);
y_global_best = func(x_global_best);

hop = ub - lb;




l=0;% Loop counter
%% 改进方式2：融合变异
% Main loop
while l<Max_iter
    for i=1:SearchAgents_no
        if rand < 1-sqrt(l/Max_iter) && num2 ~= 0  % 满足条件执行变异
            x =integration(x,i,x_global_best,num2,l,Max_iter,lb,ub);
        else
            p_y(i) = sum(w(i, :) .* x(i, :));
            p = p_y(i) - y(i);
            g_temp = p.*x(i, :);

            % Indivual Direction
            p_d = x_hist_best(i, :) - x(i, :);
            f_p_bias = y_hist_best(i) - y(i);
            p_d = p_d .* f_p_bias;
            p_d = p_d ./ ((sum(p_d)+eps).*(sum(p_d)+eps));

            d_p = p_d + g_hist_best(i, :);

            % Group Direction
            c_d = x_global_best - x(i, :);
            f_c_bias = y_global_best - y(i);
            c_d = c_d .* f_c_bias;
            c_d = c_d ./ ((sum(c_d)+eps).*(sum(c_d)+eps));

            d_g = c_d + g_global_best;

            % Gradient Estimation
            r1 = rand(1, dim);
            r2 = rand(1, dim);

            g = (1 - r1 - r2).*g_temp + r1 .* d_p + r2 .* d_g;
            g = g ./ (sum(g) + eps);

            m(i,:) = beta1.*m(i,:)+(1-beta1).*g;
            v(i,:) = beta2*v(i,:)+(1-beta2)*g.^2;
            w(i,:) = w(i,:) - m(i,:)/(sqrt(v(i,:))+eps);

            % Advice Forward
            x_o = x(i, :) + exp(-l/(0.1*Max_iter)) * 0.1 .* hop .* g;
            Flag4ub=x_o>ub;
            Flag4lb=x_o<lb;
            x_o = (x_o.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
            y_o = func(x_o);

            % Random Search
            r = random('Uniform', -pi/2, pi/2, 1, dim);
            x_n = x(i, :) + tan(r) .* hop/(1 + l) * 0.5;
            Flag4ub=x_n>ub;
            Flag4lb=x_n<lb;
            x_n = (x_n.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
            y_n = func(x_n);

            % Encircling Mechanism
            d = x_hist_best(i, :) - x(i, :);
            d_g = x_global_best - x(i, :);

            r1 = rand(1, dim);
            r2 = rand(1, dim);

            x_m = (1-r1-r2).*x(i, :) + r1.*d + r2.*d_g;
            Flag4ub=x_m>ub;
            Flag4lb=x_m<lb;
            x_m = (x_m.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
            y_m = func(x_m);


            % Discriminant Condition
            x_summary = [x_m; x_n; x_o];
            y_summary = [y_m, y_n, y_o];
            y_summary(isnan(y_summary)) = inf;
            ind = y_summary==min(y_summary);
            y_i = min(y_summary);
            x_i = x_summary(ind, :);
            x_i = x_i(1, :);

            if y_i < y(i)
                y(i) = y_i;
                x(i, :) = x_i;
                if y_i < y_hist_best(i)
                    y_hist_best(i) = y_i;
                    x_hist_best(i, :) = x_i;
                    g_hist_best(i, :) = g_temp;
                    if y_i < y_global_best
                        y_global_best = y_i;
                        x_global_best = x_i;
                        g_global_best = g_temp;
                    end
                end
            else
                if rand()<0.3
                    y(i) = y_i;
                    x(i, :) = x_i;
                end
            end
        end
    end

%% 改进方式3：尾部变异
        if num3 ~= 0
            
            for i=1:size(x,1)
                NEW_pos =integration(x,i,x_global_best,num3,l,Max_iter,lb,ub);
                Flag4ub=NEW_pos(i,:)>ub;
                Flag4lb=NEW_pos(i,:)<lb;
                NEW_pos(i,:)=(NEW_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
                fitness= fobj(NEW_pos(i,:));
                if(y(i)>fitness)
                    y(i)=fitness;
                    x(i,:) = NEW_pos(i,:);
                end
                if(y_global_best>fitness)
                    y_global_best=fitness;
                    x(i,:) = NEW_pos(i,:);
                    x_global_best=x(i,:);
                end
            end
        end



    l=l+1;
    %     fprintf("%d, %f\n", l, y_global_best)
    Convergence_curve(l) = y_global_best;

end

end

