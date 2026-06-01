
clear; clc; close all;
load('Oxford_Battery_Degradation_Dataset_1.mat'); 

cellData = Cell1;
cycleNames = fieldnames(cellData);
numCycles = length(cycleNames);


target_cycle = cycleNames{50}; 
data_50 = cellData.(target_cycle).C1dc;

t_50 = (data_50.t - data_50.t(1)) * 24 * 3600; 
v_50 = data_50.v;
T_50 = data_50.T;


capacity_array = zeros(1, numCycles);

for i = 1:numCycles
    cyc = cycleNames{i};
    try
        q = cellData.(cyc).C1dc.q;
       
        capacity_array(i) = abs(q(end) - q(1));
    catch
        capacity_array(i) = NaN; 
    end
end

fig = figure('Name','Datasetv visualisation');

subplot(2,1,1);
plot(1:numCycles, capacity_array, '-o', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 4);

yline(capacity_array(1)*0.80, ...
    '--', ...
    '80% EOL Threshold', ...
    'LineWidth', 2);

title(' Cell 1 Capacity Degradation');
xlabel('Cycle Number');
ylabel('Discharge Capacity (Ah)');
grid on;

subplot(2,1,2)

yyaxis left
p1 = plot(t_50, v_50, 'LineWidth', 2);
p1.Color = [0 0.4470 0.7410];      

ylabel('Voltage (V)')
ax = gca;
ax.YColor = p1.Color;

yyaxis right
p2 = plot(t_50, T_50, 'LineWidth', 2);
p2.Color = [0.8500 0.3250 0.0980]; 

ylabel('Temperature (°C)')
ax.YColor = p2.Color;

xlabel('Time (seconds)')
title(['Internal Dynamics (' target_cycle ')'])

grid on

sgtitle(' Battery Degradation ');
