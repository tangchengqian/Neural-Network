% LSTM_SSA_Optimized.m
% ����SSA�Ż�LSTM��������λ������Ԥ��

%% --- �û����� ---
excelFile       = '�ֽ�����ʷ���ݼ�-����.xlsx';   % ����λ�����е�Excel�ļ�
sheetName       = 'Sheet1';              % �������ڹ�����
range           = 'A1:A1500';           % λ�����ݷ�Χ
historyLen      = 10;                    % ��ʷ�������ݹ�Ԥ��ʹ�õ�ʱ�䴰���ȣ�
predictHorizon = 10;                    % δ��Ԥ�ⲽ��
trainRatio      = 0.7;                   % ѵ��������

%% --- ��ȡ��Ԥ�������� ---
data = xlsread(excelFile, sheetName, range);
data = data(~isnan(data));           % ȥ��NaN
N = length(data);
t = (1:N)';                           % ʱ������

% ����ලѧϰ���ݼ�: ÿ historyLen ��Ԥ����һ��
numSamples = N - historyLen;
X = zeros(historyLen, numSamples);
Y = zeros(1, numSamples);
for i = 1:numSamples
    X(:,i) = data(i:i+historyLen-1);
    Y(i)    = data(i+historyLen);
end

% ����ѵ�����Ͳ��Լ�
numTrain = floor(trainRatio * numSamples);
XTrain = X(:,1:numTrain);
YTrain = Y(1:numTrain);
XTest  = X(:,numTrain+1:end);
YTest  = Y(numTrain+1:end);

% ��һ�����ֱ������������ӳ��
[XTrainNorm, inputPS]   = mapminmax(XTrain,  0, 1);
XTestNorm              = mapminmax('apply', XTest,  inputPS);
[YTrainNorm, outputPS] = mapminmax(YTrain,  0, 1);

% ת������Ϊ cell ���飬ȷ��ά����ȷ
[numFeatures, numTrainSamples] = size(XTrainNorm);
[~, numTestSamples]            = size(XTestNorm);
assert(numFeatures == historyLen, '��������ά�Ȳ�ƥ�� historyLen');

XTrainCell = cell(1, numTrainSamples);
for i = 1:numTrainSamples
    XTrainCell{i} = XTrainNorm(:,i);
end
XTestCell = cell(1, numTestSamples);
for i = 1:numTestSamples
    XTestCell{i} = XTestNorm(:,i);
end

%% --- SSA �������� ---
pop       = 30;    % ��Ⱥ����
Max_iter  = 10;    % ����������
dim       = 3;     % �Ż� LSTM �ĳ�����ά��
lb        = [10, 10, 0.0001]; % �±߽�: [���ص�Ԫ, ���ѵ������, ѧϰ��]
ub        = [300, 200, 0.01];  % �ϱ߽�

% ���� fun ��Ҫ�Ķ������
numResponses = 1;  % Ԥ�����ά�ȣ�YTrainNorm �� 1����������

% ����Ŀ�꺯�� fobj(x) ���� [fitness, net]
% �� XTrainNorm ת����ƥ�� fun �е��ع��߼�������Ϊ�У�
fobj = @(x) fun(x, [], numResponses, XTrainNorm');

% ���� SSA �����Ż����������λ��������
[Best_pos, Best_score, curve, BestNet] = SSA(pop, Max_iter, lb, ub, dim, fobj);

% ���ƽ�������
figure;
plot(curve, 'r-', 'LineWidth', 2);
xlabel('��������'); ylabel('RMSE');
title('SSA-LSTM ������������'); grid on;

disp(['�������ص�Ԫ: ', num2str(round(Best_pos(1))) ]);
disp(['����ѵ������: ', num2str(round(Best_pos(2))) ]);
disp(['����ѧϰ��: ', num2str(Best_pos(3))]);

% ����������缰��һ������
save('BestNet.mat', 'BestNet', 'inputPS', 'outputPS');

%% --- ʹ����ѳ�����ѵ������ LSTM ---
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

% ����ѵ���������
netFinal = trainNetwork(XTrainCell, YTrainNorm', layers, options);

%% --- ģ��Ԥ�� ---
% ���Լ�Ԥ��
YPredTestNorm = predict(netFinal, XTestCell);
YPredTest     = mapminmax('reverse', YPredTestNorm, outputPS);

% ����Ԥ��δ��ֵ
recentWindow = data(end - historyLen + 1:end);
futurePred   = zeros(predictHorizon, 1);
for k = 1:predictHorizon
    Xn   = mapminmax('apply', recentWindow, inputPS);
    pn   = predict(netFinal, {Xn});
    pred = mapminmax('reverse', pn, outputPS);
    futurePred(k) = pred;
    recentWindow  = [recentWindow(2:end); pred];
end

%% --- ���չʾ ---
figure;
subplot(2,1,1);
plot(t(historyLen+1:end), data(historyLen+1:end), 'k', 'LineWidth',1.5);
hold on;
plot(t(numTrain+historyLen+1:end), YPredTest, 'r--', 'LineWidth',1.5);
legend('ʵ��ֵ','���Լ�Ԥ��');
title('��� SSA-LSTM ���Լ�Ԥ��'); xlabel('ʱ�䲽'); ylabel('λ��'); grid on;

subplot(2,1,2);
plot(N + (1:predictHorizon), futurePred, 'b-o', 'LineWidth',1.5);
title('����Ԥ��δ��λ��'); xlabel('ʱ�䲽'); ylabel('λ��'); grid on;

%% --- ����Ԥ���� ---
resultTable = table((N+1:N+predictHorizon)', futurePred, ...
    'VariableNames', {'TimeIndex','ForecastDisplacement'});
writetable(resultTable, 'SSA_LSTM_Forecast_Results.xlsx');
disp('Ԥ����ɣ�����ѱ��浽 SSA_LSTM_Forecast_Results.xlsx');
