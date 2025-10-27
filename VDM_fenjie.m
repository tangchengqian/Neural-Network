% ����������������
clear;
clc;

% ��һ������ȡ Excel ����
% �����ļ����͹�����
filename = '��������.xlsx'; % ���滻Ϊ���� Excel �ļ���
sheet = 1;

% ��ȡ����
data = readtable(filename, 'Sheet', sheet);

% �����һ�������ڣ��ڶ������ۻ�λ��
dates = data{:, 1};
displacements = data{:, 2};

% �ڶ�����Ԥ��������
% ������ת��Ϊ datetime ��ʽ
dates = datetime(dates, 'ConvertFrom', 'excel');

% ȷ��λ������Ϊ������
displacements = displacements(:);

% ��������ִ�� VMD �ֽ�
% ��� VMD ����·��
addpath('path_to_vmd_function'); % ���滻Ϊ VMD �������ڵ�·��

% ���� VMD ����
alpha = 2000;       % ƽ����Լ��
tau = 0;            % ʱ�䲽��
K = 3;              % ģ̬����
DC = 0;             % �Ƿ����ֱ������
init = 1;           % ��ʼ������
tol = 1e-6;         % �������

% ִ�� VMD �ֽ�
[u, ~, ~] = VMD(displacements, alpha, tau, K, DC, init, tol);

% ���Ĳ�������ģ̬����
% ����Ƶ�����ԣ���ģ̬��������Ϊ�����������������
% ��������һ��ģ̬��������ڶ�����������������������
trend = u(1, :)';
periodic = u(2, :)';
random = u(3, :)';

% ���岽������Ϳ��ӻ����
% ����������
result_table = table(dates, trend, periodic, random, ...
    'VariableNames', {'Date', 'Trend', 'Periodic', 'Random'});

% ���������µ� Excel �ļ�
writetable(result_table, 'decomposed_results.xlsx');

% ���ӻ����
figure;
subplot(4, 1, 1);
plot(dates, displacements);
title('Original Displacement');
xlabel('Date');
ylabel('Displacement');

subplot(4, 1, 2);
plot(dates, trend);
title('Trend Component');
xlabel('Date');
ylabel('Trend');

subplot(4, 1, 3);
plot(dates, periodic);
title('Periodic Component');
xlabel('Date');
ylabel('Periodic');

subplot(4, 1, 4);
plot(dates, random);
title('Random Component');
xlabel('Date');
ylabel('Random');
