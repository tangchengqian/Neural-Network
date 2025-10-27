clc;
clear 
close all

options = trainingOptions('adam', ...
    'MaxEpochs', 70, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 60, ...
    'LearnRateDropFactor', 0.2, ...
    'L2Regularization', 0.01, ...
    'ExecutionEnvironment', 'gpu', ...  % 使用 GPU 加速
    'Verbose', 0, ...
    'Plots', 'training-progress');
%% LSTM预测
tic
load origin_data.mat
load vmd_data.mat

disp('…………………………………………………………………………………………………………………………')
disp('单一的LSTM预测')
disp('…………………………………………………………………………………………………………………………')

num_samples = length(X);       % 样本个数 
kim = 1;                      % 延时步长（kim个历史数据作为自变量）
zim =  1;                      % 跨zim个时间点进行预测
or_dim = size(X,2);


%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.7;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    % 重构后的数据24输入，1个输出，共1571行，通过这个格式转换，
    % 当i=1时，把第1行的24个输入打包放进一个1*1的cell包里，
    % 以此类推共变成1571个cell，因为{i, 1}所以这1571个cell排成一列
    % 而1个cell内部的24个数是24*1竖着排列的
    % cell的数据格式是double
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  创建LSTM网络，
layers = [ ...
    sequenceInputLayer(f_)              % 输入层
    lstmLayer(70)                      
    reluLayer                           
    fullyConnectedLayer(outdim)         % 回归层
    regressionLayer];

%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 70, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 60, ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', 0.01, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'training-progress');                    % 画出曲线

%  训练
net = trainNetwork(vp_train, vt_train, layers, options);
%analyzeNetwork(net);% 查看网络结构
%  预测
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

%  数据格式转换
T_sim1 = cell2mat(T_sim1);% cell2mat将cell元胞数组转换为普通数组
T_sim2 = cell2mat(T_sim2);

% 指标计算
disp('训练集误差指标')
[mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1');
fprintf('\n')

disp('测试集误差指标')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')
toc


tic
disp('…………………………………………………………………………………………………………………………')
disp('VMD-LSTM预测')
disp('…………………………………………………………………………………………………………………………')

imf=u;
c=size(imf,1);
%% 对每个分量建模
for d=1:c
disp(['第',num2str(d),'个分量建模'])

X_imf=[X(:,1:end-1) imf(d,:)'];
num_samples = length(X_imf);  % 样本个数 

%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.7;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';


P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';


%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  创建LSTM网络，
layers = [ ...
    sequenceInputLayer(f_)              % 输入层
    lstmLayer(70)                      % LSTM层
    reluLayer                           % Relu激活层
    fullyConnectedLayer(outdim)         % 回归层
    regressionLayer];

%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 70, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 60, ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', 0.01, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'training-progress');                    % 画出曲线

%  训练
net = trainNetwork(vp_train, vt_train, layers, options);
%  预测
t_sim5 = predict(net, vp_train); 
t_sim6 = predict(net, vp_test); 

%  数据反归一化
T_sim5_imf = mapminmax('reverse', t_sim5, ps_output);
T_sim6_imf = mapminmax('reverse', t_sim6, ps_output);

%  数据格式转换
T_sim5(d,:) = cell2mat(T_sim5_imf);% cell2mat将cell元胞数组转换为普通数组
T_sim6(d,:) = cell2mat(T_sim6_imf);
T_train5(d,:)= T_train;
T_test6(d,:)= T_test;
end

% 各分量预测的结果相加
T_sim5=sum(T_sim5);
T_sim6=sum(T_sim6);
T_train5=sum(T_train5);
T_test6=sum(T_test6);

% 指标计算
disp('训练集误差指标')
[mae5,rmse5,mape5,error5]=calc_error(T_train5,T_sim5);
fprintf('\n')

disp('测试集误差指标')
[mae6,rmse6,mape6,error6]=calc_error(T_test6,T_sim6);
fprintf('\n')
toc

%% VMD-SSA-LSTM预测
tic
disp('…………………………………………………………………………………………………………………………')
disp('VMD-SSA-LSTM预测')
disp('…………………………………………………………………………………………………………………………')

% SSA参数设置
pop=30; % 种群数量
Max_iter=10; % 最大迭代次数
dim=3; % 优化LSTM的3个参数
lb = [50,50,0.001];%下边界
ub = [300,300,0.01];%上边界
% —— 不再传 numFeatures，featureDim 交给 fun.m 动态计算 ——  
numResponses = size(t_train,1);
fobj = @(x) fun(x, [], numResponses, X);
[Best_pos,Best_score,curve,BestNet]=SSA(pop,Max_iter,lb,ub,dim,fobj);
% —— 保存 BestNet 及依赖变量 —— 
save('BestNet.mat', 'BestNet', 'X', 'imf', 'num_samples', 'c', 'or_dim');

% 绘制进化曲线
figure
plot(curve,'r-','linewidth',3)
xlabel('进化代数')
ylabel('均方根误差RMSE')
legend('最佳适应度')
title('SSA-LSTM的进化收敛曲线')

disp('')
disp(['最优隐藏单元数目为   ',num2str(round(Best_pos(1)))]);
disp(['最优最大训练周期为   ',num2str(round(Best_pos(2)))]);
disp(['最优初始学习率为   ',num2str((Best_pos(3)))]);

%% 对每个分量建模
for d=1:c
disp(['第',num2str(d),'个分量建模'])

X_imf=[X(:,1:end-1) imf(d,:)'];

%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.7;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

% 最佳参数的LSTM预测
layers = [ ...
    sequenceInputLayer(f_)              % 输入层
    lstmLayer(round(Best_pos(1)))      % LSTM层
    reluLayer                           % Relu激活层
    fullyConnectedLayer(outdim)         % 回归层
    regressionLayer];


options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', round(Best_pos(2)), ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', Best_pos(3), ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', round(Best_pos(2)*0.9), ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', 0.001, ...          % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'training-progress');                    % 画出曲线

%  训练
net = trainNetwork(vp_train, vt_train, layers, options);
%  预测
t_sim7 = predict(net, vp_train); 
t_sim8 = predict(net, vp_test); 

%  数据反归一化
T_sim7_imf = mapminmax('reverse', t_sim7, ps_output);
T_sim8_imf = mapminmax('reverse', t_sim8, ps_output);

%  数据格式转换
T_sim7(d,:) = cell2mat(T_sim7_imf);% cell2mat将cell元胞数组转换为普通数组
T_sim8(d,:) = cell2mat(T_sim8_imf);
T_train7(d,:)= T_train;
T_test8(d,:)= T_test;
end

% 各分量预测的结果相加
T_sim7=sum(T_sim7);
T_sim8=sum(T_sim8);
T_train7=sum(T_train7);
T_test8=sum(T_test8);

% 指标计算
disp('训练集误差指标')
[mae7,rmse7,mape7,error7]=calc_error(T_train7,T_sim7);
fprintf('\n')

disp('测试集误差指标')
[mae8,rmse8,mape8,error8]=calc_error(T_test8,T_sim8);
fprintf('\n')
toc

%% ===== 新增代码：VMD-SSA-LSTM详细结果输出 =====
% 绘制VMD-SSA-LSTM预测结果与原数据对比图
figure;
subplot(2,1,1)
plot(T_test8, 'k', 'LineWidth', 2);
hold on;
plot(T_sim8, 'r', 'LineWidth', 2);
legend('实际值', 'VMD-SSA-LSTM预测值');
title('VMD-SSA-LSTM预测结果与实际值对比');
xlabel('样本序号');
ylabel('值');
grid on;

subplot(2,1,2)
plot(error8, 'b', 'LineWidth', 1.5);
title('VMD-SSA-LSTM预测绝对误差');
xlabel('样本序号');
ylabel('绝对误差');
grid on;

% 创建详细结果表格（使用英文变量名）
N_test = length(T_test8);
detailed_results = table(...
    (1:N_test)', ...          % Sample index
    T_test8', ...             % Actual values
    T_sim8', ...              % Predicted values
    abs(T_test8' - T_sim8'), ... % Absolute error
    'VariableNames', {'SampleIndex', 'ActualValue', 'PredictedValue', 'AbsoluteError'});

% 创建误差指标表格
metrics_table = table(...
    mae8, rmse8, mape8, ...
    'VariableNames', {'MAE', 'RMSE', 'MAPE'}, ...
    'RowNames', {'TestSetMetrics'});

% 创建模型参数表格
params_table = table(...
    round(Best_pos(1)), ...      % LSTM neurons
    round(Best_pos(2)), ...      % Max epochs
    Best_pos(3), ...             % Initial learning rate
    pop, ...                    % SSA population size
    Max_iter, ...               % SSA iterations
    'VariableNames', {'LSTM_Neurons', 'Max_Epochs', 'Initial_LearningRate', 'SSA_Population', 'SSA_Iterations'});

% 输出到Excel文件
output_filename = 'VMD_SSA_LSTM_Results.xlsx';

% 写入详细预测结果
writetable(detailed_results, output_filename, 'Sheet', 'Detailed Results');

% 写入误差指标
writetable(metrics_table, output_filename, 'Sheet', 'Performance Metrics', 'WriteRowNames', true);

% 写入模型参数
writetable(params_table, output_filename, 'Sheet', 'Model Parameters');

% 添加说明信息 - 使用兼容旧版本的方法
model_info = {
    'Model: VMD-SSA-LSTM Prediction Model';
    '1. VMD (Variational Mode Decomposition): Signal decomposition';
    '2. SSA (Sparrow Search Algorithm): Optimizes LSTM parameters';
    '3. LSTM (Long Short-Term Memory): Neural network architecture';
    '';
    'Metrics Explanation:';
    'MAE: Mean Absolute Error';
    'RMSE: Root Mean Squared Error';
    'MAPE: Mean Absolute Percentage Error (%)';
    '';
    'Data Description:';
    '1. Detailed Results: Actual values, predictions and errors for test set';
    '2. Performance Metrics: Model evaluation metrics';
    '3. Model Parameters: Optimized model parameters'
};


% 使用writetable将说明写入Excel（需要转换为表）
% 将单元格数组转换为表
model_info_table = table(model_info, 'VariableNames', {'Description'});
writetable(model_info_table, output_filename, 'Sheet', 'Description');

disp('==============================================');
disp(['VMD-SSA-LSTM results saved to: ' output_filename]);
disp('Sheets included:');
disp('1. Detailed Results - Actual values, predictions and errors');
disp('2. Performance Metrics - MAE, RMSE, MAPE');
disp('3. Model Parameters - Optimized model parameters');
disp('4. Description - Model and metrics explanation');
disp('==============================================');
%% ===== 新增代码结束 =====

%% =============== 5. VMD-SSA-LSTM模型预测新数据（修正后） ===============
disp('…………………………………………………………………………………………………………………………')
disp('VMD-SSA-LSTM模型预测新数据')
disp('…………………………………………………………………………………………………………………………')

% —— 0. 确保 BestNet 及其它必要变量已在工作区 —— 
if ~exist('BestNet','var')
    % 如果你之前已经保存过 BestNet：
    if isfile('BestNet.mat')
        S = load('BestNet.mat','BestNet','X','imf','num_samples','c','or_dim');
        BestNet    = S.BestNet;
        X          = S.X;
        imf        = S.imf;
        num_samples= S.num_samples;
        c          = S.c;
        or_dim     = S.or_dim;
    else
        error(['找不到训练好的 BestNet，请先运行 SSA-LSTM 训练部分，' ...
               '或把 BestNet 保存到 BestNet.mat 再试。']);
    end
end

% 如果你没有用 mat 文件保存，确保这一脚本和训练脚本在同一个 session 里连续运行，
% 并且训练部分产生的 BestNet、X、imf、num_samples、c、or_dim 
% 都没有被清除或覆盖。

% —— 0. 从 BestNet 输入层自动算出 kim —— 
or_dim = size(X,2);                                 % 原始特征维度（6）
inputSize = BestNet.Layers(1).InputSize;            % e.g. 35, 65, 125 …
kim = round( (inputSize - (or_dim-1)) / or_dim );   % (inputSize - 5)/6 → 5, 10, 20…

% 1. 载入新因子数据（只有 or_dim-1 列）
new_data_file = '分解后的周期数据集_预测因子.xlsx';
new_factors   = xlsread(new_data_file, '输入数据');    % 得到 N_new×(or_dim-1)

% 2. 检查列数一致
if size(new_factors, 2) ~= or_dim - 1
    error('新数据的特征维度不一致！应为 %d 列因子', or_dim - 1);
end

% 3. 获取训练集最后 kim 条 IMF 历史
last_idx = num_samples;
imf_history_buffer = zeros(c, kim);
for d = 1:c
    imf_history_buffer(d, :) = imf(d, last_idx-kim+1 : last_idx);
end

% 4. 循环递推预测
N_new = size(new_factors,1);
T_pred_components = zeros(c, N_new);


 for k = 1:N_new
    for d = 1:c
        % —— 4.1 构造因子和 IMF 历史，凑齐 kim 行 —— 
        if k <= kim
            offset = kim - (k-1);
            % 部分因子历史
            hist_factors_partial = [
                X(last_idx-offset+1:last_idx, 1:or_dim-1);
                new_factors(1:k-1, :)
            ];
            % 部分 IMF 历史（训练集末尾）
            training_part = imf(d, last_idx-offset+1:last_idx);

            % 用 if/else 代替 &&/||
            if k > 1
                predicted_part = imf_history_buffer(d, kim+1 : kim+(k-1));
            else
                predicted_part = [];
            end

            % 合并得到完整的 IMF 部分历史
            hist_imf_d_partial = [training_part, predicted_part];
        else
            % 当 k>kim 时，直接取最近 kim 条
            hist_factors_partial = new_factors(k-kim:k-1, :);
            hist_imf_d_partial   = imf_history_buffer(d, end-kim+1:end);
        end

        % —— 4.1.1 统一 pad／全量化 —— 
        num_rows = size(hist_factors_partial, 1);
        if num_rows < kim
            pad = zeros(kim - num_rows, or_dim);
            % pad 的前 or_dim-1 列给因子，最后一列给 IMF
            hist_factors_full = [pad(:,1:or_dim-1); hist_factors_partial];
            hist_imf_d_full   = [zeros(1, kim-num_rows), hist_imf_d_partial];
        else
            hist_factors_full = hist_factors_partial;
            hist_imf_d_full   = hist_imf_d_partial;
        end

        % —— 4.2 拼成矩阵并展开 —— 
        X_imf_block = [hist_factors_full, hist_imf_d_full'];   % kim×or_dim
        inputVec    = reshape(X_imf_block, 1, kim*or_dim);     % 1×(kim*or_dim)

        % —— 4.3 拼接本期因子 —— 
        curFactors  = new_factors(k, :);                       % 1×(or_dim-1)
        modelInput  = [inputVec, curFactors]';                 % inputSize×1

        % —— 4.4 预测 & 更新缓存 —— 
        pred_imf_d = predict(BestNet, modelInput);
        imf_history_buffer(d, end+1) = pred_imf_d;
        T_pred_components(d, k)      = pred_imf_d;
    end
end


% 5. 叠加所有分量并输出
T_pred_final    = sum(T_pred_components,1)';
prediction_dates = (datetime('today') + days(0:N_new-1))';
results_table    = table(prediction_dates, T_pred_final, ...
                         'VariableNames',{'PredictionDate','ForecastedValue'});
writetable(results_table, 'VMD_SSA_LSTM_Forecasts.xlsx');

figure;
plot(prediction_dates, T_pred_final, 'b-o','LineWidth',2);
title('VMD-SSA-LSTM 递推预测结果');
xlabel('日期'); ylabel('预测累计位移');
grid on; datetick('x','yyyy-mm-dd','keepticks');

disp('预测完成，结果已保存在 VMD_SSA_LSTM_Forecasts.xlsx');
