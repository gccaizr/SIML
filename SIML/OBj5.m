function classLoss=OBj5(x,input,output,M)
input(:,x(1:M)<0.5)=[];                             % 特征筛选工作
if isempty(input)
    classLoss=1;
else
    rng(1);                                        % 固定随机数（为了重现）
    SVMModel = fitcsvm(input,output,'Standardize',true,'KernelFunction','RBF',...
        'KernelScale',x(M+1)); % 径向基核训练 SVM 分类器
    CVSVMModel = crossval(SVMModel,'KFold',5);    % 交叉验证 5折SVM 分类器
    classLoss = kfoldLoss(CVSVMModel);             % 计算交叉验证误差
end
end


