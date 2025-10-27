% LSTM_Forecast_NoDecomp.m
% ������ʷ�����λ����Ϣ��LSTMԤ�⣬����VMD�ֽ�
% ��ȡExcel��λ��ʱ�����У�ѵ��LSTMģ�Ͳ�����Ԥ��δ��ֵ

%% --- �û����� ---
excelFile       = '�ֽ�����ʷ���ݼ�.xlsx';   % ����λ�����е�Excel�ļ�
sheetName       = 'Sheet1';              % �������ڹ�����
range           = 'A1:A1500';           % λ�����ݷ�Χ
historyLen      = 400;                    % ��ʷ�������ݹ�Ԥ��ʹ�õ�ʱ�䴰���ȣ�
predictHorizon = 100;                    % δ��Ԥ�ⲽ��
trainRatio      = 0.7;                   % ѵ��������

%% --- ��ȡ��Ԥ�������� ---
data = xlsread(excelFile, sheetName, range);
data = data(~isnan(data));           % ȥ��NaN
N = length(data);

t = (1:N)';                           % ʱ������

% ����ලѧϰ���ݼ�: ÿhistoryLen��Ԥ����һ��
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
[XTrainNorm, inputPS]  = mapminmax(XTrain,  0, 1);
XTestNorm             = mapminmax('apply', XTest,  inputPS);
[YTrainNorm, outputPS]= mapminmax(YTrain,  0, 1);

% ת������Ϊcell���飬��Ӧʹ����ֵ������ʽ
XTrainCell = cell(size(XTrainNorm,2),1);
for i = 1:size(XTrainNorm,2)
    XTrainCell{i} = XTrainNorm(:,i);
end
XTestCell = cell(size(XTestNorm,2),1);
for i = 1:size(XTestNorm,2)
    XTestCell{i} = XTestNorm(:,i);
end

%% --- ������ѵ��LSTM���� ---
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

% ѵ�����磬YTrainNormΪ��ֵ����
net = trainNetwork(XTrainCell, YTrainNorm', layers, options);

%% --- ģ��Ԥ�� ---
% 1) ���Լ�Ԥ��
YPredTestNorm = predict(net, XTestCell);            % 1��numTest
YPredTest     = mapminmax('reverse', YPredTestNorm,    outputPS);

% 2) ����Ԥ��δ��predictHorizon��
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

%% --- ���չʾ ---
figure;
subplot(2,1,1);
plot(t(historyLen+1:end), data(historyLen+1:end),'k','LineWidth',1.5);
hold on;
plot(t(numTrain+historyLen+1:end), YPredTest,'r--','LineWidth',1.5);
legend('ʵ��ֵ','���Լ�Ԥ��');
title('LSTM���Լ�Ԥ��');
xlabel('ʱ�䲽'); ylabel('λ��'); grid on;

subplot(2,1,2);
plot(N+(1:predictHorizon), futurePred,'b-o','LineWidth',1.5);
title('����Ԥ��δ��λ��');
xlabel('ʱ�䲽'); ylabel('λ��'); grid on;

%% --- ����Ԥ���� ---
resultTable = table((N+1:N+predictHorizon)', futurePred, ...
    'VariableNames',{'TimeIndex','ForecastDisplacement'});
writetable(resultTable, 'LSTM_Forecast_Results.xlsx');
disp('Ԥ����ɣ�����ѱ��浽LSTM_Forecast_Results.xlsx');
