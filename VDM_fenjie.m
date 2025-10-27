% 清除工作区和命令窗口
clear;
clc;

% 第一步：读取 Excel 数据
% 设置文件名和工作表
filename = '测试数据.xlsx'; % 请替换为您的 Excel 文件名
sheet = 1;

% 读取数据
data = readtable(filename, 'Sheet', sheet);

% 假设第一列是日期，第二列是累积位移
dates = data{:, 1};
displacements = data{:, 2};

% 第二步：预处理数据
% 将日期转换为 datetime 格式
dates = datetime(dates, 'ConvertFrom', 'excel');

% 确保位移数据为列向量
displacements = displacements(:);

% 第三步：执行 VMD 分解
% 添加 VMD 函数路径
addpath('path_to_vmd_function'); % 请替换为 VMD 函数所在的路径

% 设置 VMD 参数
alpha = 2000;       % 平滑性约束
tau = 0;            % 时间步长
K = 3;              % 模态数量
DC = 0;             % 是否包含直流分量
init = 1;           % 初始化方法
tol = 1e-6;         % 误差容限

% 执行 VMD 分解
[u, ~, ~] = VMD(displacements, alpha, tau, K, DC, init, tol);

% 第四步：分类模态分量
% 根据频率特性，将模态分量分类为趋势项、周期项和随机项
% 这里假设第一个模态是趋势项，第二个是周期项，第三个是随机项
trend = u(1, :)';
periodic = u(2, :)';
random = u(3, :)';

% 第五步：保存和可视化结果
% 创建结果表格
result_table = table(dates, trend, periodic, random, ...
    'VariableNames', {'Date', 'Trend', 'Periodic', 'Random'});

% 保存结果到新的 Excel 文件
writetable(result_table, 'decomposed_results.xlsx');

% 可视化结果
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
