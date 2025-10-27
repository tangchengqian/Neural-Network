% LSTM_Forecast_NoDecomp.m
% 基于历史随机性位移信息的LSTM预测，无需VMD分解
% 读取Excel中位移时间序列，训练LSTM模型并递推预测未来值

%% --- 用户参数 ---
excelFile       = '分解后的历史数据集.xlsx';   % 包含位移序列的Excel文件
sheetName       = 'Sheet1';              % 数据所在工作表
range           = 'A1:A1500';           % 位移数据范围
historyLen      = 400;                    % 历史步长（递归预测使用的时间窗长度）
predictHorizon = 100;                    % 未来预测步数
trainRatio      = 0.7;                   % 训练集比例

%% --- 读取并预处理数据 ---
data = xlsread(excelFile, sheetName, range);
data = data(~isnan(data));           % 去除NaN
N = length(data);

t = (1:N)';                           % 时间索引

% 构造监督学习数据集: 每historyLen步预测下一步
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
[XTrainNorm, inputPS]  = mapminmax(XTrain,  0, 1);
XTestNorm             = mapminmax('apply', XTest,  inputPS);
[YTrainNorm, outputPS]= mapminmax(YTrain,  0, 1);

% 转换输入为cell数组，响应使用数值向量形式
XTrainCell = cell(size(XTrainNorm,2),1);
for i = 1:size(XTrainNorm,2)
    XTrainCell{i} = XTrainNorm(:,i);
end
XTestCell = cell(size(XTestNorm,2),1);
for i = 1:size(XTestNorm,2)
    XTestCell{i} = XTestNorm(:,i);
end

%% --- 创建并训练LSTM网络 ---
featureDim = historyLen;
layers = [
    sequenceInputLayer(featureDim)
    lstmLayer(50,'OutputMode','last')
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',50, ...
    'LearnRateDropFactor',0.5, ...
    'GradientThreshold',1, ...
    'Verbose',0, ...
    'Plots','training-progress');

% 训练网络，YTrainNorm为数值向量
net = trainNetwork(XTrainCell, YTrainNorm', layers, options);

%% --- 模型预测 ---
% 1) 测试集预测
YPredTestNorm = predict(net, XTestCell);            % 1×numTest
YPredTest     = mapminmax('reverse', YPredTestNorm,    outputPS);

% 2) 递推预测未来predictHorizon步
recentWindow = data(end-historyLen+1:end);
futurePred   = zeros(predictHorizon,1);
for k = 1:predictHorizon
    Xnorm    = mapminmax('apply', recentWindow, inputPS);
    Xcell    = {Xnorm};
    predNorm = predict(net, Xcell);                  % scalar
    pred     = mapminmax('reverse', predNorm, outputPS);  % scalar
    futurePred(k) = pred;
    recentWindow = [recentWindow(2:end); pred];
end

%% --- 结果展示 ---
figure;
subplot(2,1,1);
plot(t(historyLen+1:end), data(historyLen+1:end),'k','LineWidth',1.5);
hold on;
plot(t(numTrain+historyLen+1:end), YPredTest,'r--','LineWidth',1.5);
legend('实际值','测试集预测');
title('LSTM测试集预测');
xlabel('时间步'); ylabel('位移'); grid on;

subplot(2,1,2);
plot(N+(1:predictHorizon), futurePred,'b-o','LineWidth',1.5);
title('递推预测未来位移');
xlabel('时间步'); ylabel('位移'); grid on;

%% --- 保存预测结果 ---
resultTable = table((N+1:N+predictHorizon)', futurePred, ...
    'VariableNames',{'TimeIndex','ForecastDisplacement'});
writetable(resultTable, 'LSTM_Forecast_Results.xlsx');
disp('预测完成，结果已保存到LSTM_Forecast_Results.xlsx');
