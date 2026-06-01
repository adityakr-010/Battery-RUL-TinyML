clc;
clear;
close all;

load('Oxford_Battery_Degradation_Dataset_1.mat');

cellData = Cell2;

cycleNames = fieldnames(cellData);

numCycles = length(cycleNames);

%Features 

capacityVec          = zeros(numCycles,1);
energyVec            = zeros(numCycles,1);
durationVec          = zeros(numCycles,1);

avgTempVec           = zeros(numCycles,1);
maxTempVec           = zeros(numCycles,1);

voltageMeanVec       = zeros(numCycles,1);
voltageStdVec        = zeros(numCycles,1);

dQdVPeakVec          = zeros(numCycles,1);
dQdVAreaVec          = zeros(numCycles,1);

entropyVec           = zeros(numCycles,1);

kneePositionVec      = zeros(numCycles,1);

validCycles          = [];

for i = 1:numCycles

    try

        cycName = cycleNames{i};

        if ~isfield(cellData.(cycName),'C1dc')
            continue;
        end

        data = cellData.(cycName).C1dc;

        t = data.t;
        v = data.v;
        q = data.q;
        T = data.T;

        idx = ~(isnan(t) | isnan(v) | isnan(q) | isnan(T));

        t = t(idx);
        v = v(idx);
        q = q(idx);
        T = T(idx);


        if length(v) < 50
            continue;
        end

        t = t - t(1);
        t = t * 24 * 3600;

        durationVec(i) = t(end);

        capacityVec(i) = abs(q(end) - q(1));

        energyVec(i) = trapz(t, abs(v .* q));

        avgTempVec(i) = mean(T);

        maxTempVec(i) = max(T);

        voltageMeanVec(i) = mean(v);

        voltageStdVec(i) = std(v);

        % dQ/dV FEATURE

        dq = diff(q);
        dv = diff(v);

        dv(abs(dv) < 1e-5) = 1e-5;

        dQdV = dq ./ dv;

        dQdV = smoothdata(dQdV,'movmean',15);

        dQdVPeakVec(i) = max(abs(dQdV));

        dQdVAreaVec(i) = trapz(abs(dQdV));

        % ENTROPY
        
        v_norm = normalize(v);

        p = abs(v_norm);
        p = p / sum(p);

        entropyVec(i) = -sum(p .* log(p + 1e-12));

        % KNEE DETECTION

        dvdt = gradient(v);

        [~, kneeIdx] = min(dvdt);

        kneePositionVec(i) = kneeIdx / length(v);

        validCycles = [validCycles; i];

    catch

        continue;

    end

end

capacityVec     = capacityVec(validCycles);
energyVec       = energyVec(validCycles);
durationVec     = durationVec(validCycles);

avgTempVec      = avgTempVec(validCycles);
maxTempVec      = maxTempVec(validCycles);

voltageMeanVec  = voltageMeanVec(validCycles);
voltageStdVec   = voltageStdVec(validCycles);

dQdVPeakVec     = dQdVPeakVec(validCycles);
dQdVAreaVec     = dQdVAreaVec(validCycles);

entropyVec      = entropyVec(validCycles);

kneePositionVec = kneePositionVec(validCycles);

%RUL Target

N = length(capacityVec);

RUL = (N:-1:1)';
RUL = RUL - 1;



FeatureMatrix = [
    capacityVec,...
    energyVec,...
    durationVec,...
    avgTempVec,...
    maxTempVec,...
    voltageMeanVec,...
    voltageStdVec,...
    dQdVPeakVec,...
    dQdVAreaVec,...
    entropyVec,...
    kneePositionVec
];


featureNames = {
    'Capacity'
    'Energy'
    'Duration'
    'AvgTemp'
    'MaxTemp'
    'VoltageMean'
    'VoltageStd'
    'dQdVPeak'
    'dQdVArea'
    'Entropy'
    'KneePosition'
};


FeatureMatrix = normalize(FeatureMatrix);


figure('Color','k',...
       'Name',' Degradation Features',...
       'Position',[100 100 1200 800])

subplot(3,3,1)
plot(FeatureMatrix(:,1),'LineWidth',2)
title('Capacity','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(3,3,2)
plot(FeatureMatrix(:,2),'LineWidth',2)
title('Energy','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(3,3,3)
plot(FeatureMatrix(:,3),'LineWidth',2)
title('Duration','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(3,3,4)
plot(RUL,'LineWidth',2)
title('RUL Target','Color','w')
xlabel('Cycle','Color','w')
ylabel('Remaining Cycles','Color','w')
grid on

set(findall(gcf,'Type','axes'),...
    'Color','k',...
    'XColor','w',...
    'YColor','w',...
    'FontSize',12)

sgtitle('Core Battery Degradation Features',...
    'Color','w',...
    'FontSize',22)


subplot(3,3,5)
plot(FeatureMatrix(:,4),'LineWidth',2)
title('Average Temperature','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(3,3,6)
plot(FeatureMatrix(:,5),'LineWidth',2)
title('Maximum Temperature','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(3,3,7)
plot(FeatureMatrix(:,6),'LineWidth',2)
title('Voltage Mean','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(3,3,8)
plot(FeatureMatrix(:,7),'LineWidth',2)
title('Voltage Standard Deviation','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

set(findall(gcf,'Type','axes'),...
    'Color','k',...
    'XColor','w',...
    'YColor','w',...
    'FontSize',12)

sgtitle('Features ',...
    'Color','w',...
    'FontSize',22)

% FIGURE 3 - ELECTROCHEMICAL FEATURES

figure('Color','k',...
       'Name','Electrochemical Features',...
       'Position',[200 200 1200 800])

subplot(2,2,1)
plot(FeatureMatrix(:,8),'LineWidth',2)
title('dQ/dV Peak','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(2,2,2)
plot(FeatureMatrix(:,9),'LineWidth',2)
title('dQ/dV Area','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(2,2,3)
plot(FeatureMatrix(:,10),'LineWidth',2)
title('Entropy','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

subplot(2,2,4)
plot(FeatureMatrix(:,11),'LineWidth',2)
title('Knee Position','Color','w')
xlabel('Cycle','Color','w')
ylabel('Normalized Value','Color','w')
grid on

set(findall(gcf,'Type','axes'),...
    'Color','k',...
    'XColor','w',...
    'YColor','w',...
    'FontSize',12)

sgtitle('Electrochemical Aging Features',...
    'Color','w',...
    'FontSize',22)

% FEATURE CORRELATION WITH RUL

corrScores = zeros(size(FeatureMatrix,2),1);

for k = 1:size(FeatureMatrix,2)

    c = corrcoef(FeatureMatrix(:,k),RUL);

    corrScores(k) = abs(c(1,2));

end



figure('Color','k',...
       'Name','Feature Correlation with RUL',...
       'Position',[250 250 1200 700])

bar(corrScores,'LineWidth',2)

grid on

xticks(1:length(featureNames))
xticklabels(featureNames)
xtickangle(45)

ylabel('Absolute Correlation','Color','w')

title('Feature Correlation with RUL',...
      'Color','w',...
      'FontSize',22)

set(gca,...
    'Color','k',...
    'XColor','w',...
    'YColor','w',...
    'FontSize',12)

% CORRELATION MATRIX

corrMat = corrcoef([FeatureMatrix RUL]);

allNames = [featureNames; {'RUL'}];

figure('Color','k',...
       'Name','Feature Correlation Matrix',...
       'Position',[100 100 1000 900])

h = heatmap( ...
    allNames,...
    allNames,...
    corrMat);

h.Title = 'Feature Correlation Matrix';
h.Colormap = parula;
h.ColorLimits = [-1 1];

h.CellLabelFormat = '%.2f';

h.FontSize = 12;
h.XDisplayLabels = allNames;
h.YDisplayLabels = allNames;

fprintf('Valid Cycles Extracted : %d\n',N);

fprintf('Feature Dimensions     : %d x %d\n',...
    size(FeatureMatrix,1),...
    size(FeatureMatrix,2));
