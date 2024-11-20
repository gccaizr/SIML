
clear
clc
close all
Function_name='F8'; % 测试函数编号（F1~F23）
[lb,ub,dim,fobj]=Get_Functions_details(Function_name); % 获取目标函数对应参数
SearchAgents_no=30; %种群规模
Max_iteration=100; %最大迭代次数

%% 模型训练
for i = 1:30   % 次数可以修改运行次数
 disp(['第',num2str(i),'次实验']);
num1 = 2;  % 头部混沌变异改进方案选择：1-10，tent、Logistic、Cubic、chebyshev、Piecewise、sinusoidal、Sine,ICMIC, Circle,Bernoulli
num2 = 8; % 身体融合变异改进方案选择：1-15，高斯变异，t分布扰动，自适应t分布，柯西变异扰动，差分变异扰动，高斯随机游走，莱维飞行,三角形游走,螺旋飞行，黄金正弦，正余弦,透镜成像反向学习,纵横交叉，动态反向学习，随机游走
num3 = 10;  % 尾部拼接变异改进方法选择：
[Best_score(i,:),Best_pos(i,:),cg_curve(i,:)]=ESOA(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);      % 遗传优化算法
num1 = 0;  % 无改进
num2 = 0;  % 无改进
num3 = 0;  % 无改进
[Best_score1(i,:),Best_pos1(i,:),cg_curve1(i,:)]=ESOA(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);  % 粒子群优化算法
end

%% 结果绘图
figure('Position',[500 500 660 290])
% 绘制搜索空间
subplot(1,2,1);
func_plot(Function_name);
title('Test function')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])
grid off

%绘制收敛曲线
CNT=20;   % 绘制图形随机选点的数量(可自行设置)
k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
iter=1:1:Max_iteration;
Cg_curve=mean(cg_curve);
Cg_curve1=mean(cg_curve1);
subplot(1,2,2);
semilogy(iter(k),Cg_curve(k),'-*','LineWidth',1)
hold on
semilogy(iter(k),Cg_curve1(k),'-p','LineWidth',1)
legend('改进后','原始算法')
grid off
xlabel('迭代次数')
ylabel('目标函数值')
title('不同优化算法的进化曲线对比图')

%% 输出结果
disp('======改进后算法结果==========');
display(['改进后算法30次实验最优适应度值(Best) : ', num2str(min(Best_score))]);
display(['改进后算法30次实验最优解对应的平均适应度值(mean) : ', num2str(mean(Best_score))]);
display(['改进后算法30次实验最差适应度值(wrost) : ', num2str(max(Best_score))]);
display(['改进后算法30次实验标准差（std） : ', num2str(std(Best_score))]);


disp('======该进前算法结果============');
display(['该进前算法30次实验最优适应度值(Best) : ', num2str(min(Best_score1))]);
display(['该进前算法30次实验最优解对应的平均适应度值(mean) : ', num2str(mean(Best_score1))]);
display(['该进前算法30次实验最差适应度值(wrost) : ', num2str(max(Best_score1))]);
display(['该进前算法30次实验标准差（std） : ', num2str(std(Best_score1))]);



