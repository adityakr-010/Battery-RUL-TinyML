clear; clc; close all;

load('Oxford_Battery_Degradation_Dataset_1.mat');

cellData = Cell1;

%  cycle
cycleName = 'cyc0000';
data = cellData.(cycleName).C1dc;

t = data.t;
v = data.v;
q = data.q;
T = data.T;

t = (t - t(1)) * 24 * 3600;

figure('Name','Oxford Battery Cycle Analysis');

subplot(2,2,1)
plot(t,'LineWidth',1.5)
title('Elapsed Time')
xlabel('Sample Index')
ylabel('Time (s)')
grid on

subplot(2,2,2)
plot(t,v,'LineWidth',1.5)
title('Voltage Profile')
xlabel('Time (s)')
ylabel('Voltage (V)')
grid on

subplot(2,2,3)
plot(t,q,'LineWidth',1.5)
title('Capacity / Charge')
xlabel('Time (s)')
ylabel('Charge (Ah)')
grid on

subplot(2,2,4)
plot(t,T,'LineWidth',1.5)
title('Temperature Profile')
xlabel('Time (s)')
ylabel('Temperature (°C)')
grid on

sgtitle(['Oxford Dataset - ' cycleName])