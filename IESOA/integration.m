function Positions = integration(Positions,i,Alpha_pos,num2,l,iter,lb,ub)
N =size(Positions,1); % 计算种群数量
dim=size(Positions,2); % 计算维度
switch num2

    case 1                                  % 高斯变异
        sigma = 0.7; %高斯变异参数
        Positions(i,:)=Positions(i,:).* Gaussian(Positions(i,:),Alpha_pos,sigma);
    case 2                                  % t分布扰动
        Positions(i,:)=  Alpha_pos+ trnd(l)*Alpha_pos;
    case 3
        freen = exp(4.*(l/iter).^2);        % 自适应t分布
        Positions(i,:)=  Alpha_pos+ trnd(freen)*Alpha_pos;  %t分布公式
    case 4                                  % 柯西变异扰动

        cauch = cauchyrnd(0,1,[N,1]); % 生成1个柯西随机数
        Positions(i,:)=  Alpha_pos+ cauch(i)*Alpha_pos; % 柯西分布公式
    case 5                                  % 差分变异扰动
        A=randperm(N);
        A(A==i)=[];
        a=A(1);
        Positions(i,:)=rand*(Alpha_pos-Positions(i,:))-rand.*(Positions(a,:)-Positions(i,:));
    case 6                                  % 高斯随机游走
        sigma = 0.9;  %高斯变异参数，这个大家可以自行更改
        a0 = Gaussian(Positions(i,:),Alpha_pos,sigma);
        stepsize=a0.*(Positions(i,:)-Alpha_pos);
        Positions(i,:) =Positions(i,:)+stepsize.*randn(size(Positions(i,:))); %在随机游走基础上添加高斯
    case 7                                  % 莱维飞行
        RL = 0.15*levy(N,dim,1.5); %这里的1.5就是莱维飞行的beta，一般就取1.5。也可以自行更改
        L = Alpha_pos-Positions(i,:);
        Positions(i,: )  = Positions(i,: )+L.*RL(i,:);

    case 8                                   % 三角形游走

        rg=0.1-((0.1)*l/iter);
        r=rand*rg;
        L = Alpha_pos-Positions(i,:);
        LP = L.*rand(1,dim);
        alph = L.*L+LP.*LP-2*LP.*L.*cos(2*pi*rand(1,dim));
        Positions(i,:)  = Positions(i,:)+r*alph;%(rand-1)

    case 9                                   % 螺旋飞行
        k=1; %螺旋飞行步长
        L = 2*rand-1;
        z = exp(k*cos(pi*(1-(l/iter))));
        Positions(i,:)= Alpha_pos+exp(z*L)*cos(2*pi*L)*abs(Alpha_pos-Positions(i,:));
    case 10                                  % 黄金正弦
        a=-pi;
        b=pi;
        gold=double((sqrt(5)-1)/2);      % golden proportion coefficient, around 0.618
        x1=a+(1-gold)*(b-a);
        x2=a+gold*(b-a);
        r=rand;
        r1=(2*pi)*r;
        r2=r*pi;
        for vv = 1:dim % in j-th dimension
            Positions(i,vv) = Positions(i,vv)*abs(sin(r1)) - r2*sin(r1)*abs(x1*Alpha_pos(1,vv)-x2*Positions(1,vv));   %黄金正弦计算公式
        end
    case 11                                   % 正余弦
        r1 = 2*(1-l/iter);
        r2 = 2*pi*rand;r3 =2*rand;
        if rand >0.5
            for vv = 1:dim
                Positions(i,vv) = Positions(i,vv)+r1*sin(r2)*abs(r3*Alpha_pos(vv)-Positions(i,vv));
            end
        else
            for vv = 1:dim
                Positions(i,vv) = Positions(i,vv)+r1*cos(r2)*abs(r3*Alpha_pos(vv)-Positions(i,vv));
            end
        end
    case 12                                  % 透镜成像反向学习
        k=(1+(l/iter)^0.5)^10;
        Positions(i,:) = (ub+lb)/2+(ub+lb)/(2*k)-Positions(i,:)/k;
    case 13                                  % 纵横交叉
        if (mod(i,2) == 1)
            for  j = 1 : dim
                r = rand();
                c = 2*rand()-1;
                Positions(i,j) = r*Positions(i,j)+(1-r)*Positions(i+1,j)+c*(Positions(i,j)-Positions(i+1,j));
            end
        else
            for j = 1 : dim
                r = rand();
                c = 2*rand()-1;
                Positions(i,j) = r * Positions(i-1,j)+(1-r)*Positions(i,j)+c*(Positions(i-1,j)-Positions(i,j));
            end
        end
    case 14                                     %  动态反向学习
        Xss = max(Positions,[],1); %求每一维度的最大值
        Xxx = min(Positions,[],1); %求每一维度的最小值
        for h1=1:size(Positions,1)
            for h2 = 1:size(Positions,2)
                Positions(h1,h2) = Xss(h2) + Xxx(h2) - Positions(h1,h2);
            end
        end
        for n1=1:dim
            for m1=1:size(Positions,1)
                if (Positions(m1,n1)<Xxx(n1)) || (Positions(m1,n1)>Xss(n1))
                    Positions(m1,n1) = rand*(Xss(n1)-Xxx(n1))+Xxx(n1);%超出边界重新赋值
                end
            end
        end
    case 15                    %  随机游走
        min_c = min(Positions);  % 随机游走策略用到的参数
        max_d = max(Positions);
        Positions(i,:) =(Positions(i,:)-lb).*(max_d-min_c)./(ub-lb)+min_c;
end
end


%% 需要用到的函数
%% 高斯变异
function [y] = Gaussian(x,mu,sigma)
y = 1/(sqrt(2*pi)*sigma)*exp(-(x-mu).^2/(2*sigma^2));
end

%%  柯西随机数
function r= cauchyrnd(varargin)
a=	0.0;
b=	1.0;
n=	1;
% Check the arguments
if(nargin >= 1)
    a=	varargin{1};
    if(nargin >= 2)
        b=			varargin{2};
        b(b <= 0)=	NaN;	% Make NaN of out of range values.
        if(nargin >= 3),	n=	[varargin{3:end}];		end
    end
end
% Generate
r=	cauchyinv(rand(n), a, b);
end

function x= cauchyinv(p, varargin)
% Default values
a=	0.0;
b=	1.0;
% Check the arguments
if(nargin >= 2)
    a=	varargin{1};
    if(nargin == 3)
        b=			varargin{2};
        b(b <= 0)=	NaN;	% Make NaN of out of range values.
    end
end
if((nargin < 1) || (nargin > 3))
    error('At least one argument, at most three!');
end
p(p < 0 | 1 < p)=	NaN;
% Calculate
x=			a + b.*tan(pi*(p-0.5));
% Extreme values.
if(numel(p) == 1), 	p= repmat(p, size(x));		end
x(p == 0)=	-Inf;
x(p == 1)=	Inf;
end

%% 莱维飞行
function [z] = levy(n,m,beta)
num = gamma(1+beta)*sin(pi*beta/2); % used for Numerator
den = gamma((1+beta)/2)*beta*2^((beta-1)/2); % used for Denominator
sigma_u = (num/den)^(1/beta);% Standard deviation
u = random('Normal',0,sigma_u,n,m);
v = random('Normal',0,1,n,m);
z =u./(abs(v).^(1/beta));
end
