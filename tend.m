% =============================================================================
% 滑动窗口最小二乘法趋势项预测脚本 (Matlab 2018b)
%
% 功能：
%   1. 从 Excel 文件读取历史数据（假设第一列是 x，第二列是 y）。
%   2. 对长度为 w 的滑动窗口使用最小二乘法进行线性回归，得到斜率 b(k) 和截距 a(k)。
%   3. 计算每个时刻 i（i ≥ w）的拟合值 y_fit(i) = a(i?w+1) + b(i?w+1)*x(i)。
%   4. 使用最后一个窗口的回归系数对未来 N_pred 个点进行预测（外推）。
%   5. 绘制原始值、拟合值和预测值。
%   6. 将“时间 x”、“原始 y”、“拟合 y”、“预测 y”写入新的 Excel 文件。
%
% 说明：
%   ? 请自行修改 “fileName” 为你的历史数据 Excel 文件名（含路径）。
%   ? 窗口长度 w、未来预测步数 N_pred 均可在脚本开头灵活修改。
%   ? 预测时假设 x 序列是等间隔的数值，如果实际 x 不等间隔，请自行调整 x_pred 的生成方法。
% =============================================================================

%% ---- 用户可修改区 ----
fileName = 'tend-历史数据.xlsx';   % Excel 文件名
sheetName = 'Sheet1';         % 读取工作表名称
w = 10;                       
N_pred = 10;                 
outputFile = '趋势项预测结果.xlsx';  
%% ---- 结束用户可修改区 ----

%% 1. 读取历史数据，并检查列数
data = xlsread(fileName, sheetName);
if size(data,2) < 2
    error('从 Excel 读取的数据只有 %d 列，至少需要两列 (x 和 y)。请检查文件或 sheetName。', size(data,2));
end

x = data(:,1);
y = data(:,2);
N = length(x);

if N < w
    error('数据长度 (%d) 小于窗口长度 w (%d)。请增大数据或减小 w。', N, w);
end

%% 2. 初始化
% 拟合值序列，前 w?1 点无法拟合，填 NaN
y_fit = NaN(N,1);

% 用来存储每个窗口的回归系数
a_vals = NaN(N-w+1,1);   % 截距 a(k)，对应窗口 k=1 对应数据行 i=w
b_vals = NaN(N-w+1,1);   % 斜率 b(k)

%% 3. 滑动窗口回归
for k = 1 : (N - w + 1)
    % 当前窗口 [k : k+w-1]
    idx = k : (k + w - 1);
    xw = x(idx);
    yw = y(idx);
    
    % 3.1 计算均值
    xm = mean(xw);
    ym = mean(yw);
    
    % 3.2 计算协方差与方差
    Cov_xy = sum( (xw - xm) .* (yw - ym) ) / w;
    Var_x  = sum( (xw - xm).^2 ) / w;
    
    % 3.3 计算回归系数 b 和 a
    b = Cov_xy / Var_x;
    a = ym - b * xm;
    
    a_vals(k) = a;
    b_vals(k) = b;
    
    % 3.4 计算当前窗口末尾点 (索引 i = k+w-1) 的拟合值
    i = k + w - 1;
    y_fit(i) = a + b * x(i);
end

%% 4. 使用最后一个窗口系数向后预测 N_pred 点
% 最后一个窗口的系数
a_last = a_vals(end);
b_last = b_vals(end);

% 假设 x 是等间隔的，计算步长 Δx
if N >= 2
    dx = x(end) - x(end-1);
else
    dx = 1;  % 如果仅有一个点，则默认间隔为 1
end

% 生成未来 N_pred 个 x_pred
x_pred = x(end) + dx*(1:N_pred)';  % 列向量

% 预测 y_pred = a_last + b_last * x_pred
y_pred = a_last + b_last * x_pred;

%% 5. 合并“原始数据、拟合数据、预测数据”，以便绘图和导出
% 构造完整的时间序列向量 x_all
x_all = [ x; x_pred ];             % (N+N_pred) × 1

% 构造对应的 y_all，将拟合值和预测值合并
y_all = [ y, y_fit ];              % N×2

% 对未来 N_pred 行，原始 y 置 NaN，拟合 y 置 NaN，预测 y 置 y_pred
y_future = NaN(N_pred,1);
y_fit_future = NaN(N_pred,1);
y_all = [ y_all; [y_future, y_fit_future] ];   % (N+N_pred)×2

% 合并所有输出列
% 列 1: x_all
% 列 2: 原始 y（前 N 行有效，后 N_pred 行 NaN）
% 列 3: 拟合 y（前 N 行第 2 列是拟合值，后 N_pred 行 NaN）
% 列 4: 预测 y（前 N 行 NaN，后 N_pred 行为预测值）
pred_col = [ NaN(N,1); y_pred ];
output_matrix = [ x_all, y_all, pred_col ];  % (N+N_pred)×4

%% 6. 绘图
figure;
hold on;
% 原始 y
plot(x, y, 'k.-', 'DisplayName','原始值');
% 拟合 y（仅从 w 到 N 有值）
plot(x, y_fit, 'b.-', 'DisplayName','拟合值');
% 预测 y
plot(x_pred, y_pred, 'r.-', 'DisplayName','预测值');

legend('Location','best');
xlabel('x');
ylabel('y');
title(sprintf('滑动窗口最小二乘趋势拟合 (w=%d)，向后预测 %d 步', w, N_pred));
grid on;
hold off;

%% 7. 将结果输出到 Excel
% 拼写输出列标题
header = {'x','y_original','y_fitted','y_forecast'};
% 将数据矩阵写入 Excel （从第二行开始），第一行写标题
xlswrite(outputFile, header, 'Sheet1','A1');
xlswrite(outputFile, output_matrix, 'Sheet1','A2');

fprintf('=========================================\n');
fprintf('趋势项预测完成，结果已保存至: %s\n', outputFile);
fprintf('Sheet1 包含以下列：\n');
fprintf('  列 A: x\n');
fprintf('  列 B: 原始 y\n');
fprintf('  列 C: 拟合 y\n');
fprintf('  列 D: 预测 y\n');
fprintf('=========================================\n');
