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
    'ExecutionEnvironment', 'gpu', ...  % ʹ�� GPU ����
    'Verbose', 0, ...
    'Plots', 'training-progress');
%% LSTMԤ��
tic
load origin_data.mat
load vmd_data.mat

disp('��������������������������������������������������������������������������������������������')
disp('��һ��LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')

num_samples = length(X);       % �������� 
kim = 1;                      % ��ʱ������kim����ʷ������Ϊ�Ա�����
zim =  1;                      % ��zim��ʱ������Ԥ��
or_dim = size(X,2);


%  �ع����ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,:)];
end


% ѵ�����Ͳ��Լ�����
outdim = 1;                                  % ���һ��Ϊ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  ��ʽת��
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    % �ع��������24���룬1���������1571�У�ͨ�������ʽת����
    % ��i=1ʱ���ѵ�1�е�24���������Ž�һ��1*1��cell���
    % �Դ����ƹ����1571��cell����Ϊ{i, 1}������1571��cell�ų�һ��
    % ��1��cell�ڲ���24������24*1�������е�
    % cell�����ݸ�ʽ��double
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  ����LSTM���磬
layers = [ ...
    sequenceInputLayer(f_)              % �����
    lstmLayer(70)                      
    reluLayer                           
    fullyConnectedLayer(outdim)         % �ع��
    regressionLayer];

%  ��������
options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', 70, ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', 0.01, ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', 60, ...                   % ѵ��850�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.2, ...                    % ѧϰ�ʵ�������
    'L2Regularization', 0.01, ...         % ���򻯲���
    'ExecutionEnvironment', 'cpu',...                 % ѵ������
    'Verbose', 0, ...                                 % �ر��Ż�����
    'Plots', 'training-progress');                    % ��������

%  ѵ��
net = trainNetwork(vp_train, vt_train, layers, options);
%analyzeNetwork(net);% �鿴����ṹ
%  Ԥ��
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

%  ���ݸ�ʽת��
T_sim1 = cell2mat(T_sim1);% cell2mat��cellԪ������ת��Ϊ��ͨ����
T_sim2 = cell2mat(T_sim2);

% ָ�����
disp('ѵ�������ָ��')
[mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1');
fprintf('\n')

disp('���Լ����ָ��')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')
toc


tic
disp('��������������������������������������������������������������������������������������������')
disp('VMD-LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')

imf=u;
c=size(imf,1);
%% ��ÿ��������ģ
for d=1:c
disp(['��',num2str(d),'��������ģ'])

X_imf=[X(:,1:end-1) imf(d,:)'];
num_samples = length(X_imf);  % �������� 

%  �ع����ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% ѵ�����Ͳ��Լ�����
outdim = 1;                                  % ���һ��Ϊ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';


P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';


%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  ��ʽת��
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  ����LSTM���磬
layers = [ ...
    sequenceInputLayer(f_)              % �����
    lstmLayer(70)                      % LSTM��
    reluLayer                           % Relu�����
    fullyConnectedLayer(outdim)         % �ع��
    regressionLayer];

%  ��������
options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', 70, ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', 0.01, ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', 60, ...                   % ѵ��850�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.2, ...                    % ѧϰ�ʵ�������
    'L2Regularization', 0.01, ...         % ���򻯲���
    'ExecutionEnvironment', 'cpu',...                 % ѵ������
    'Verbose', 0, ...                                 % �ر��Ż�����
    'Plots', 'training-progress');                    % ��������

%  ѵ��
net = trainNetwork(vp_train, vt_train, layers, options);
%  Ԥ��
t_sim5 = predict(net, vp_train); 
t_sim6 = predict(net, vp_test); 

%  ���ݷ���һ��
T_sim5_imf = mapminmax('reverse', t_sim5, ps_output);
T_sim6_imf = mapminmax('reverse', t_sim6, ps_output);

%  ���ݸ�ʽת��
T_sim5(d,:) = cell2mat(T_sim5_imf);% cell2mat��cellԪ������ת��Ϊ��ͨ����
T_sim6(d,:) = cell2mat(T_sim6_imf);
T_train5(d,:)= T_train;
T_test6(d,:)= T_test;
end

% ������Ԥ��Ľ�����
T_sim5=sum(T_sim5);
T_sim6=sum(T_sim6);
T_train5=sum(T_train5);
T_test6=sum(T_test6);

% ָ�����
disp('ѵ�������ָ��')
[mae5,rmse5,mape5,error5]=calc_error(T_train5,T_sim5);
fprintf('\n')

disp('���Լ����ָ��')
[mae6,rmse6,mape6,error6]=calc_error(T_test6,T_sim6);
fprintf('\n')
toc

%% VMD-SSA-LSTMԤ��
tic
disp('��������������������������������������������������������������������������������������������')
disp('VMD-SSA-LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')

% SSA��������
pop=30; % ��Ⱥ����
Max_iter=10; % ����������
dim=3; % �Ż�LSTM��3������
lb = [50,50,0.001];%�±߽�
ub = [300,300,0.01];%�ϱ߽�
% ���� ���ٴ� numFeatures��featureDim ���� fun.m ��̬���� ����  
numResponses = size(t_train,1);
fobj = @(x) fun(x, [], numResponses, X);
[Best_pos,Best_score,curve,BestNet]=SSA(pop,Max_iter,lb,ub,dim,fobj);
% ���� ���� BestNet ���������� ���� 
save('BestNet.mat', 'BestNet', 'X', 'imf', 'num_samples', 'c', 'or_dim');

% ���ƽ�������
figure
plot(curve,'r-','linewidth',3)
xlabel('��������')
ylabel('���������RMSE')
legend('�����Ӧ��')
title('SSA-LSTM�Ľ�����������')

disp('')
disp(['�������ص�Ԫ��ĿΪ   ',num2str(round(Best_pos(1)))]);
disp(['�������ѵ������Ϊ   ',num2str(round(Best_pos(2)))]);
disp(['���ų�ʼѧϰ��Ϊ   ',num2str((Best_pos(3)))]);

%% ��ÿ��������ģ
for d=1:c
disp(['��',num2str(d),'��������ģ'])

X_imf=[X(:,1:end-1) imf(d,:)'];

%  �ع����ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% ѵ�����Ͳ��Լ�����
outdim = 1;                                  % ���һ��Ϊ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  ��ʽת��
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

% ��Ѳ�����LSTMԤ��
layers = [ ...
    sequenceInputLayer(f_)              % �����
    lstmLayer(round(Best_pos(1)))      % LSTM��
    reluLayer                           % Relu�����
    fullyConnectedLayer(outdim)         % �ع��
    regressionLayer];


options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', round(Best_pos(2)), ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', Best_pos(3), ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', round(Best_pos(2)*0.9), ...                   % ѵ��850�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.2, ...                    % ѧϰ�ʵ�������
    'L2Regularization', 0.001, ...          % ���򻯲���
    'ExecutionEnvironment', 'cpu',...                 % ѵ������
    'Verbose', 0, ...                                 % �ر��Ż�����
    'Plots', 'training-progress');                    % ��������

%  ѵ��
net = trainNetwork(vp_train, vt_train, layers, options);
%  Ԥ��
t_sim7 = predict(net, vp_train); 
t_sim8 = predict(net, vp_test); 

%  ���ݷ���һ��
T_sim7_imf = mapminmax('reverse', t_sim7, ps_output);
T_sim8_imf = mapminmax('reverse', t_sim8, ps_output);

%  ���ݸ�ʽת��
T_sim7(d,:) = cell2mat(T_sim7_imf);% cell2mat��cellԪ������ת��Ϊ��ͨ����
T_sim8(d,:) = cell2mat(T_sim8_imf);
T_train7(d,:)= T_train;
T_test8(d,:)= T_test;
end

% ������Ԥ��Ľ�����
T_sim7=sum(T_sim7);
T_sim8=sum(T_sim8);
T_train7=sum(T_train7);
T_test8=sum(T_test8);

% ָ�����
disp('ѵ�������ָ��')
[mae7,rmse7,mape7,error7]=calc_error(T_train7,T_sim7);
fprintf('\n')

disp('���Լ����ָ��')
[mae8,rmse8,mape8,error8]=calc_error(T_test8,T_sim8);
fprintf('\n')
toc

%% ===== �������룺VMD-SSA-LSTM��ϸ������ =====
% ����VMD-SSA-LSTMԤ������ԭ���ݶԱ�ͼ
figure;
subplot(2,1,1)
plot(T_test8, 'k', 'LineWidth', 2);
hold on;
plot(T_sim8, 'r', 'LineWidth', 2);
legend('ʵ��ֵ', 'VMD-SSA-LSTMԤ��ֵ');
title('VMD-SSA-LSTMԤ������ʵ��ֵ�Ա�');
xlabel('�������');
ylabel('ֵ');
grid on;

subplot(2,1,2)
plot(error8, 'b', 'LineWidth', 1.5);
title('VMD-SSA-LSTMԤ��������');
xlabel('�������');
ylabel('�������');
grid on;

% ������ϸ������ʹ��Ӣ�ı�������
N_test = length(T_test8);
detailed_results = table(...
    (1:N_test)', ...          % Sample index
    T_test8', ...             % Actual values
    T_sim8', ...              % Predicted values
    abs(T_test8' - T_sim8'), ... % Absolute error
    'VariableNames', {'SampleIndex', 'ActualValue', 'PredictedValue', 'AbsoluteError'});

% �������ָ����
metrics_table = table(...
    mae8, rmse8, mape8, ...
    'VariableNames', {'MAE', 'RMSE', 'MAPE'}, ...
    'RowNames', {'TestSetMetrics'});

% ����ģ�Ͳ������
params_table = table(...
    round(Best_pos(1)), ...      % LSTM neurons
    round(Best_pos(2)), ...      % Max epochs
    Best_pos(3), ...             % Initial learning rate
    pop, ...                    % SSA population size
    Max_iter, ...               % SSA iterations
    'VariableNames', {'LSTM_Neurons', 'Max_Epochs', 'Initial_LearningRate', 'SSA_Population', 'SSA_Iterations'});

% �����Excel�ļ�
output_filename = 'VMD_SSA_LSTM_Results.xlsx';

% д����ϸԤ����
writetable(detailed_results, output_filename, 'Sheet', 'Detailed Results');

% д�����ָ��
writetable(metrics_table, output_filename, 'Sheet', 'Performance Metrics', 'WriteRowNames', true);

% д��ģ�Ͳ���
writetable(params_table, output_filename, 'Sheet', 'Model Parameters');

% ���˵����Ϣ - ʹ�ü��ݾɰ汾�ķ���
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


% ʹ��writetable��˵��д��Excel����Ҫת��Ϊ��
% ����Ԫ������ת��Ϊ��
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
%% ===== ����������� =====

%% =============== 5. VMD-SSA-LSTMģ��Ԥ�������ݣ������� ===============
disp('��������������������������������������������������������������������������������������������')
disp('VMD-SSA-LSTMģ��Ԥ��������')
disp('��������������������������������������������������������������������������������������������')

% ���� 0. ȷ�� BestNet ��������Ҫ�������ڹ����� ���� 
if ~exist('BestNet','var')
    % �����֮ǰ�Ѿ������ BestNet��
    if isfile('BestNet.mat')
        S = load('BestNet.mat','BestNet','X','imf','num_samples','c','or_dim');
        BestNet    = S.BestNet;
        X          = S.X;
        imf        = S.imf;
        num_samples= S.num_samples;
        c          = S.c;
        or_dim     = S.or_dim;
    else
        error(['�Ҳ���ѵ���õ� BestNet���������� SSA-LSTM ѵ�����֣�' ...
               '��� BestNet ���浽 BestNet.mat ���ԡ�']);
    end
end

% �����û���� mat �ļ����棬ȷ����һ�ű���ѵ���ű���ͬһ�� session ���������У�
% ����ѵ�����ֲ����� BestNet��X��imf��num_samples��c��or_dim 
% ��û�б�����򸲸ǡ�

% ���� 0. �� BestNet ������Զ���� kim ���� 
or_dim = size(X,2);                                 % ԭʼ����ά�ȣ�6��
inputSize = BestNet.Layers(1).InputSize;            % e.g. 35, 65, 125 ��
kim = round( (inputSize - (or_dim-1)) / or_dim );   % (inputSize - 5)/6 �� 5, 10, 20��

% 1. �������������ݣ�ֻ�� or_dim-1 �У�
new_data_file = '�ֽ����������ݼ�_Ԥ������.xlsx';
new_factors   = xlsread(new_data_file, '��������');    % �õ� N_new��(or_dim-1)

% 2. �������һ��
if size(new_factors, 2) ~= or_dim - 1
    error('�����ݵ�����ά�Ȳ�һ�£�ӦΪ %d ������', or_dim - 1);
end

% 3. ��ȡѵ������� kim �� IMF ��ʷ
last_idx = num_samples;
imf_history_buffer = zeros(c, kim);
for d = 1:c
    imf_history_buffer(d, :) = imf(d, last_idx-kim+1 : last_idx);
end

% 4. ѭ������Ԥ��
N_new = size(new_factors,1);
T_pred_components = zeros(c, N_new);


 for k = 1:N_new
    for d = 1:c
        % ���� 4.1 �������Ӻ� IMF ��ʷ������ kim �� ���� 
        if k <= kim
            offset = kim - (k-1);
            % ����������ʷ
            hist_factors_partial = [
                X(last_idx-offset+1:last_idx, 1:or_dim-1);
                new_factors(1:k-1, :)
            ];
            % ���� IMF ��ʷ��ѵ����ĩβ��
            training_part = imf(d, last_idx-offset+1:last_idx);

            % �� if/else ���� &&/||
            if k > 1
                predicted_part = imf_history_buffer(d, kim+1 : kim+(k-1));
            else
                predicted_part = [];
            end

            % �ϲ��õ������� IMF ������ʷ
            hist_imf_d_partial = [training_part, predicted_part];
        else
            % �� k>kim ʱ��ֱ��ȡ��� kim ��
            hist_factors_partial = new_factors(k-kim:k-1, :);
            hist_imf_d_partial   = imf_history_buffer(d, end-kim+1:end);
        end

        % ���� 4.1.1 ͳһ pad��ȫ���� ���� 
        num_rows = size(hist_factors_partial, 1);
        if num_rows < kim
            pad = zeros(kim - num_rows, or_dim);
            % pad ��ǰ or_dim-1 �и����ӣ����һ�и� IMF
            hist_factors_full = [pad(:,1:or_dim-1); hist_factors_partial];
            hist_imf_d_full   = [zeros(1, kim-num_rows), hist_imf_d_partial];
        else
            hist_factors_full = hist_factors_partial;
            hist_imf_d_full   = hist_imf_d_partial;
        end

        % ���� 4.2 ƴ�ɾ���չ�� ���� 
        X_imf_block = [hist_factors_full, hist_imf_d_full'];   % kim��or_dim
        inputVec    = reshape(X_imf_block, 1, kim*or_dim);     % 1��(kim*or_dim)

        % ���� 4.3 ƴ�ӱ������� ���� 
        curFactors  = new_factors(k, :);                       % 1��(or_dim-1)
        modelInput  = [inputVec, curFactors]';                 % inputSize��1

        % ���� 4.4 Ԥ�� & ���»��� ���� 
        pred_imf_d = predict(BestNet, modelInput);
        imf_history_buffer(d, end+1) = pred_imf_d;
        T_pred_components(d, k)      = pred_imf_d;
    end
end


% 5. �������з��������
T_pred_final    = sum(T_pred_components,1)';
prediction_dates = (datetime('today') + days(0:N_new-1))';
results_table    = table(prediction_dates, T_pred_final, ...
                         'VariableNames',{'PredictionDate','ForecastedValue'});
writetable(results_table, 'VMD_SSA_LSTM_Forecasts.xlsx');

figure;
plot(prediction_dates, T_pred_final, 'b-o','LineWidth',2);
title('VMD-SSA-LSTM ����Ԥ����');
xlabel('����'); ylabel('Ԥ���ۼ�λ��');
grid on; datetick('x','yyyy-mm-dd','keepticks');

disp('Ԥ����ɣ�����ѱ����� VMD_SSA_LSTM_Forecasts.xlsx');
