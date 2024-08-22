%% T1 Post and T2 Training W/ CAMO TRANSFER LEARNING (REVISION TRAINING)
% Training a new network on T1 Post and T2 data, which now includes data from Pedram CDs. Therefore, Astro, oligoastro, oligodendro, and normal
% MRIs are all either T1 Post contrast or T2. No meningioma included because of non-standardized imaging yet.
% CAMO TRANSFER LEARNING APPLIED NOW, TTEST COMPARISON AT BOTTOM

% Loading previously trained networks
load('T1Net.mat');
load('T2Net.mat');

% Load camo net for camo TL
load('camo_net2.mat');

% Load training & testing sets (which are tied to trained nets)
load('T1Training_ds.mat');
load('T2Training_ds.mat');
load('t1test.mat');
load('t2test.mat');
load('T1Testing_ds.mat');
load('T2Testing_ds.mat');

% Pre-TL Accuracy check
% Preds1 = classify(T1Net,T1Testing_ds);
% truetest1 = t1test.Labels;
% nnz(Preds1 == truetest1)/numel(Preds1) % T1: 83.54
% 
% Preds2 = classify(T2Net,T2Testing_ds);
% truetest2 = t2test.Labels;
% nnz(Preds2 == truetest2)/numel(Preds2)
% confusionchart(truetest2,Preds2); % T2: 86.90

% Transferring Layers
CamoLayers = camo_net2.Layers;
CamoLayers(end-2) = fullyConnectedLayer(4);
CamoLayers(end) = classificationLayer;

% Setting Training Options
trainOpts1 = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'ValidationData',...
T1Training_ds,'ValidationFrequency',6,'Shuffle','every-epoch','MaxEpochs',24,'Plots','training-progress'); % T1 data

trainOpts2 = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'ValidationData',...
T2Training_ds,'ValidationFrequency',6,'Shuffle','every-epoch','MaxEpochs',24,'Plots','training-progress'); % T2 data

% Transfer Learning Training
[ExpT1Net, info] = trainNetwork(T1Training_ds, CamoLayers, trainOpts1);
[ExpT2Net, info] = trainNetwork(T2Training_ds, CamoLayers, trainOpts2);

% Testing the Networks
Preds1 = classify(ExpT1Net,T1Testing_ds);
truetest1 = t1test.Labels;
nnz(Preds1 == truetest1)/numel(Preds1)
confusionchart(truetest1,Preds1);
% Trials:
% 84.81
% 86.08
% 84.81
% 82.28
% 91.14
% 91.14 (saved)
% 89.87
% 89.87
% Mean: 87.5

Preds2 = classify(ExpT2Net,T2Testing_ds);
truetest2 = t2test.Labels;
nnz(Preds2 == truetest2)/numel(Preds2)
confusionchart(truetest2,Preds2);
% Trials:
% 92.26
% 93.45
% 90.48
% 89.88
% 88.10
% 92.86
% 94.64 (saved)
% 93.45
% 94.64
% Mean: 92.04

%% ExpT1 and ExpT2 feature spaces

load('ExpT1Net.mat');
load('ExpT2Net.mat');

load('T1Testing_ds.mat');
load('T2Testing_ds.mat');

layer = 'fc';
ExpT1NetfeaturesTest = activations(ExpT1Net,T1Testing_ds,layer,'OutputAs','rows');
ExpT2NetfeaturesTest = activations(ExpT2Net,T2Testing_ds,layer,'OutputAs','rows');

% FCBearImg = imread('CamoNet FC Bear (cropped).png'); %% SAVE THIS CODE FOR IMPLEMENTING DDI IMAGES LATER
% FCBearImg = imresize(FCBearImg,[227 227]);
% FCCanineImg = imread('CamoNet FC Canine (cropped).png');
% FCCanineImg = imresize(FCCanineImg,[227 227]);
% FCFrogImg = imread('CamoNet FC Frog (cropped).png');
% FCFrogImg = imresize(FCFrogImg,[227 227]);
% FCReptileImg = imread('CamoNet FC Reptile (cropped).png');
% FCReptileImg = imresize(FCReptileImg,[227 227]);

% Reducing Dimensionality
[coeff1,score1] = pca(ExpT1NetfeaturesTest);
[coeff2,score2] = pca(ExpT2NetfeaturesTest);
ExpT1Net_Acts = score1(:,1:2);
ExpT2Net_Acts = score2(:,1:2);

% Assigning Tumor Clusters
T1Astro = ExpT1Net_Acts(1:22,:);
T1Normal = ExpT1Net_Acts(23:58,:);
T1OA = ExpT1Net_Acts(59:66,:);
T1OD = ExpT1Net_Acts(67:79,:);

T2Astro = ExpT2Net_Acts(1:43,:);
T2Normal = ExpT2Net_Acts(44:77,:);
T2OA = ExpT2Net_Acts(78:122,:);
T2OD = ExpT2Net_Acts(123:168,:);

% Plotting Tumor Group Clusters
figure;
hold on
plot(T1Astro(:,1),T1Astro(:,2),'b*')
plot(T1Normal(:,1),T1Normal(:,2),'black*')
plot(T1OA(:,1),T1OA(:,2),'r*')
plot(T1OD(:,1),T1OD(:,2),'g*')
legend('Astrocytoma', 'Normal', 'Oligoastrocytoma','Oligodendroglioma')

figure;
hold on
plot(T2Astro(:,1),T2Astro(:,2),'bd')
plot(T2Normal(:,1),T2Normal(:,2),'blackd')
plot(T2OA(:,1),T2OA(:,2),'rd')
plot(T2OD(:,1),T2OD(:,2),'gd')
legend('Astrocytoma', 'Normal', 'Oligoastrocytoma','Oligodendroglioma')

%% Ttest

% Pre-TL accuracies
%T1
% Trials:
% 88.61
% 82.88
% 84.81
% 88.61
% 87.34
% 87.34
% 84.81 
% 83.54 (saved)
% Mean: 85.99

% T2
% Trials:
% 83.93
% 88.69
% 84.52
% 70.83
% 88.10
% 88.69 
% 79.17
% 86.90 (saved)
% Mean: 83.85

% Creating accuracy vectors
T1_scores = [88.61, 82.88, 84.81, 88.61, 87.34, 87.34, 84.81, 83.54];
T2_scores = [83.93, 88.69, 84.52, 70.83, 88.10, 88.69, 79.17, 86.90];
ExpT1_scores = [84.81, 86.08, 84.81, 82.28, 91.14, 91.14, 89.87, 89.87];
ExpT2_scores = [92.26, 93.45, 90.48, 89.88, 88.10, 92.86, 94.64, 94.64];

[h,p,ci,stats] = ttest2(T1_scores,ExpT1_scores) % p = 0.3153, h = 0
[h,p,ci,stats] = ttest2(T2_scores,ExpT2_scores) % p = 0.0035, h = 1


% ExpT1Net
% Trials:
% 84.81
% 86.08
% 84.81
% 82.28
% 91.14
% 91.14 (saved)
% 89.87
% 89.87
% Mean: 87.5

%ExpT2Net
% Trials:
% 92.26
% 93.45
% 90.48
% 89.88
% 88.10
% 92.86
% 94.64 (saved)
% 93.45
% 94.64
% Mean: 92.20







