% =============================================================================
% ����������С���˷�������Ԥ��ű� (Matlab 2018b)
%
% ���ܣ�
%   1. �� Excel �ļ���ȡ��ʷ���ݣ������һ���� x���ڶ����� y����
%   2. �Գ���Ϊ w �Ļ�������ʹ����С���˷��������Իع飬�õ�б�� b(k) �ͽؾ� a(k)��
%   3. ����ÿ��ʱ�� i��i �� w�������ֵ y_fit(i) = a(i?w+1) + b(i?w+1)*x(i)��
%   4. ʹ�����һ�����ڵĻع�ϵ����δ�� N_pred �������Ԥ�⣨���ƣ���
%   5. ����ԭʼֵ�����ֵ��Ԥ��ֵ��
%   6. ����ʱ�� x������ԭʼ y��������� y������Ԥ�� y��д���µ� Excel �ļ���
%
% ˵����
%   ? �������޸� ��fileName�� Ϊ�����ʷ���� Excel �ļ�������·������
%   ? ���ڳ��� w��δ��Ԥ�ⲽ�� N_pred �����ڽű���ͷ����޸ġ�
%   ? Ԥ��ʱ���� x �����ǵȼ������ֵ�����ʵ�� x ���ȼ���������е��� x_pred �����ɷ�����
% =============================================================================

%% ---- �û����޸��� ----
fileName = 'tend-��ʷ����.xlsx';   % Excel �ļ���
sheetName = 'Sheet1';         % ��ȡ����������
w = 10;                       
N_pred = 10;                 
outputFile = '������Ԥ����.xlsx';  
%% ---- �����û����޸��� ----

%% 1. ��ȡ��ʷ���ݣ����������
data = xlsread(fileName, sheetName);
if size(data,2) < 2
    error('�� Excel ��ȡ������ֻ�� %d �У�������Ҫ���� (x �� y)�������ļ��� sheetName��', size(data,2));
end

x = data(:,1);
y = data(:,2);
N = length(x);

if N < w
    error('���ݳ��� (%d) С�ڴ��ڳ��� w (%d)�����������ݻ��С w��', N, w);
end

%% 2. ��ʼ��
% ���ֵ���У�ǰ w?1 ���޷���ϣ��� NaN
y_fit = NaN(N,1);

% �����洢ÿ�����ڵĻع�ϵ��
a_vals = NaN(N-w+1,1);   % �ؾ� a(k)����Ӧ���� k=1 ��Ӧ������ i=w
b_vals = NaN(N-w+1,1);   % б�� b(k)

%% 3. �������ڻع�
for k = 1 : (N - w + 1)
    % ��ǰ���� [k : k+w-1]
    idx = k : (k + w - 1);
    xw = x(idx);
    yw = y(idx);
    
    % 3.1 �����ֵ
    xm = mean(xw);
    ym = mean(yw);
    
    % 3.2 ����Э�����뷽��
    Cov_xy = sum( (xw - xm) .* (yw - ym) ) / w;
    Var_x  = sum( (xw - xm).^2 ) / w;
    
    % 3.3 ����ع�ϵ�� b �� a
    b = Cov_xy / Var_x;
    a = ym - b * xm;
    
    a_vals(k) = a;
    b_vals(k) = b;
    
    % 3.4 ���㵱ǰ����ĩβ�� (���� i = k+w-1) �����ֵ
    i = k + w - 1;
    y_fit(i) = a + b * x(i);
end

%% 4. ʹ�����һ������ϵ�����Ԥ�� N_pred ��
% ���һ�����ڵ�ϵ��
a_last = a_vals(end);
b_last = b_vals(end);

% ���� x �ǵȼ���ģ����㲽�� ��x
if N >= 2
    dx = x(end) - x(end-1);
else
    dx = 1;  % �������һ���㣬��Ĭ�ϼ��Ϊ 1
end

% ����δ�� N_pred �� x_pred
x_pred = x(end) + dx*(1:N_pred)';  % ������

% Ԥ�� y_pred = a_last + b_last * x_pred
y_pred = a_last + b_last * x_pred;

%% 5. �ϲ���ԭʼ���ݡ�������ݡ�Ԥ�����ݡ����Ա��ͼ�͵���
% ����������ʱ���������� x_all
x_all = [ x; x_pred ];             % (N+N_pred) �� 1

% �����Ӧ�� y_all�������ֵ��Ԥ��ֵ�ϲ�
y_all = [ y, y_fit ];              % N��2

% ��δ�� N_pred �У�ԭʼ y �� NaN����� y �� NaN��Ԥ�� y �� y_pred
y_future = NaN(N_pred,1);
y_fit_future = NaN(N_pred,1);
y_all = [ y_all; [y_future, y_fit_future] ];   % (N+N_pred)��2

% �ϲ����������
% �� 1: x_all
% �� 2: ԭʼ y��ǰ N ����Ч���� N_pred �� NaN��
% �� 3: ��� y��ǰ N �е� 2 �������ֵ���� N_pred �� NaN��
% �� 4: Ԥ�� y��ǰ N �� NaN���� N_pred ��ΪԤ��ֵ��
pred_col = [ NaN(N,1); y_pred ];
output_matrix = [ x_all, y_all, pred_col ];  % (N+N_pred)��4

%% 6. ��ͼ
figure;
hold on;
% ԭʼ y
plot(x, y, 'k.-', 'DisplayName','ԭʼֵ');
% ��� y������ w �� N ��ֵ��
plot(x, y_fit, 'b.-', 'DisplayName','���ֵ');
% Ԥ�� y
plot(x_pred, y_pred, 'r.-', 'DisplayName','Ԥ��ֵ');

legend('Location','best');
xlabel('x');
ylabel('y');
title(sprintf('����������С����������� (w=%d)�����Ԥ�� %d ��', w, N_pred));
grid on;
hold off;

%% 7. ���������� Excel
% ƴд����б���
header = {'x','y_original','y_fitted','y_forecast'};
% �����ݾ���д�� Excel ���ӵڶ��п�ʼ������һ��д����
xlswrite(outputFile, header, 'Sheet1','A1');
xlswrite(outputFile, output_matrix, 'Sheet1','A2');

fprintf('=========================================\n');
fprintf('������Ԥ����ɣ�����ѱ�����: %s\n', outputFile);
fprintf('Sheet1 ���������У�\n');
fprintf('  �� A: x\n');
fprintf('  �� B: ԭʼ y\n');
fprintf('  �� C: ��� y\n');
fprintf('  �� D: Ԥ�� y\n');
fprintf('=========================================\n');
