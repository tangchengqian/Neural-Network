% LSTM_SSA_Optimized.m
% 基于SSA优化LSTM超参数的位移序列预测

%% --- 用户参数 ---
excelFile       = '分解后的历史数据集-测试.xlsx';   % 包含位移序列的Excel文件
sheetName       = 'Sheet1';              % 数据所在工作表
range           = 'A1:A1500';           % 位移数据范围
historyLen      = 10;                    % 历史步长（递归预测使用的时间窗长度）
predictHorizon = 10;                    % 未来预测步数
trainRatio      = 0.7;                   % 训练集比例

%% --- 读取并预处理数据 ---
data = xlsread(excelFile, sheetName, range);
data = data(~isnan(data));           % 去除NaN
N = length(data);
t = (1:N)';                           % 时间索引

% 构造监督学习数据集: 每 historyLen 步预测下一步
numSamples = N - historyLen;
X = zeros(historyLen, numSamples);
Y = zeros(1, numSamples);
for i = 1:numSamples
    X(:,i) = data(i:i+historyLen-1);
    Y(i)    = data(i+historyLen);
end

% 划分训练集和测试集
numTrain = floor(trainRatio * numSamples);
XTrain = X(:,1:numTrain);
YTrain = Y(1:numTrain);
XTest  = X(:,numTrain+1:end);
YTest  = Y(numTrain+1:end);

% 归一化：分别对输入和输出做映射
[XTrainNorm, inputPS]   = mapminmax(XTrain,  0, 1);
XTestNorm              = mapminmax('apply', XTest,  inputPS);
[YTrainNorm, outputPS] = mapminmax(YTrain,  0, 1);

% 转换输入为 cell 数组，确保维度正确
[numFeatures, numTrainSamples] = size(XTrainNorm);
[~, numTestSamples]            = size(XTestNorm);
assert(numFeatures == historyLen, '输入特征维度不匹配 historyLen');

XTrainCell = cell(1, numTrainSamples);
for i = 1:numTrainSamples
    XTrainCell{i} = XTrainNorm(:,i);
end
XTestCell = cell(1, numTestSamples);
for i = 1:numTestSamples
    XTestCell{i} = XTestNorm(:,i);
end

%% --- SSA 参数设置 ---
pop       = 30;    % 种群数量
Max_iter  = 10;    % 最大迭代次数
dim       = 3;     % 优化 LSTM 的超参数维度
lb        = [10, 10, 0.0001]; % 下边界: [隐藏单元, 最大训练周期, 学习率]
ub        = [300, 200, 0.01];  % 上边界

% 设置 fun 需要的额外参数
numResponses = 1;  % 预测输出维度（YTrainNorm 是 1×样本数）

% 定义目标函数 fobj(x) 返回 [fitness, net]
% 将 XTrainNorm 转置以匹配 fun 中的重构逻辑（样本为行）
fobj = @(x) fun(x, [], numResponses, XTrainNorm');

% 调用 SSA 进行优化，返回最佳位置与网络
[Best_pos, Best_score, curve, BestNet] = SSA(pop, Max_iter, lb, ub, dim, fobj);

% 绘制进化曲线
figure;
plot(curve, 'r-', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('RMSE');
title('SSA-LSTM 进化收敛曲线'); grid on;

disp(['最优隐藏单元: ', num2str(round(Best_pos(1))) ]);
disp(['最优训练周期: ', num2str(round(Best_pos(2))) ]);
disp(['最优学习率: ', num2str(Best_pos(3))]);

% 保存最佳网络及归一化参数
save('BestNet.mat', 'BestNet', 'inputPS', 'outputPS');

%% --- 使用最佳超参数训练最终 LSTM ---
hiddenUnits = round(Best_pos(1));
maxEpochs   = round(Best_pos(2));
initLR      = Best_pos(3);

layers = [
    sequenceInputLayer(historyLen)
    lstmLayer(hiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'InitialLearnRate', initLR, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', ceil(maxEpochs/2), ...
    'LearnRateDropFactor', 0.5, ...
    'GradientThreshold', 1, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% 重新训练最佳网络
netFinal = trainNetwork(XTrainCell, YTrainNorm', layers, options);

%% --- 模型预测 ---
% 测试集预测
YPredTestNorm = predict(netFinal, XTestCell);
YPredTest     = mapminmax('reverse', YPredTestNorm, outputPS);

% 递推预测未来值
recentWindow = data(end - historyLen + 1:end);
futurePred   = zeros(predictHorizon, 1);
for k = 1:predictHorizon
    Xn   = mapminmax('apply', recentWindow, inputPS);
    pn   = predict(netFinal, {Xn});
    pred = mapminmax('reverse', pn, outputPS);
    futurePred(k) = pred;
    recentWindow  = [recentWindow(2:end); pred];
end

%% --- 结果展示 ---
figure;
subplot(2,1,1);
plot(t(historyLen+1:end), data(historyLen+1:end), 'k', 'LineWidth',1.5);
hold on;
plot(t(numTrain+historyLen+1:end), YPredTest, 'r--', 'LineWidth',1.5);
legend('实际值','测试集预测');
title('最佳 SSA-LSTM 测试集预测'); xlabel('时间步'); ylabel('位移'); grid on;

subplot(2,1,2);
plot(N + (1:predictHorizon), futurePred, 'b-o', 'LineWidth',1.5);
title('递推预测未来位移'); xlabel('时间步'); ylabel('位移'); grid on;

%% --- 保存预测结果 ---
resultTable = table((N+1:N+predictHorizon)', futurePred, ...
    'VariableNames', {'TimeIndex','ForecastDisplacement'});
writetable(resultTable, 'SSA_LSTM_Forecast_Results.xlsx');
disp('预测完成，结果已保存到 SSA_LSTM_Forecast_Results.xlsx');
