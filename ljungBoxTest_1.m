% ljungBoxTest.m
% This script reads time series data from an Excel file and conducts a Ljung-Box white noise test.
% Requires Econometrics Toolbox (lbqtest).

% --- User Parameters ---
excelFile = '分解后的随机数据集-测试.xlsx';    % Excel file name (with extension)
sheetName = 'Sheet1';      % Sheet name or index
range = 'A:A';             % Range of the data in Excel (e.g., 'A:A' for entire column A)
lags = 20;                 % Number of lags for Ljung-Box test

% --- Read Data ---
% Option 1: Using readmatrix (R2019a+)
try
    data = readmatrix(excelFile, 'Sheet', sheetName, 'Range', range);
catch
    % Fallback for older versions
    [num, ~, ~] = xlsread(excelFile, sheetName, range);
    data = num;
end

data = data(~isnan(data));  % Remove NaNs

% --- Perform Ljung-Box Test ---
% h = test decision (1 reject null of white noise)
% pValue = p-value of the test
% QStat = test statistic
[h, pValue, QStat] = lbqtest(data, 'Lags', lags);

% --- Display Results ---
disp('Ljung-Box Test Results');
disp('------------------------');
fprintf('Number of Observations: %d\n', length(data));
fprintf('Number of Lags: %d\n', lags);
fprintf('Test Statistic (Q): %.4f\n', QStat);
fprintf('p-value: %.4f\n', pValue);
if h == 0
    disp('Result: Fail to reject null hypothesis (data is white noise)');
else
    disp('Result: Reject null hypothesis (data is NOT white noise)');
end

% --- Optional: Write Results to Excel ---
outputSheet = 'Results';
resultsTable = table(length(data), lags, QStat, pValue, h, ...
    'VariableNames', {'N', 'Lags', 'QStat', 'pValue', 'RejectH0'});

try
    writetable(resultsTable, excelFile, 'Sheet', outputSheet, 'WriteRowNames', false);
    fprintf('Results written to sheet "%s" in %s\n', outputSheet, excelFile);
catch
    warning('Could not write results to Excel.');
end
