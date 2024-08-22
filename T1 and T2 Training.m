%% T1 Post and T2 Training 
% Training a new network on T1 Post and T2 data, which now includes data from VA Hospital CDs. Therefore, Astro, oligoastro, oligodendro, and normal
% MRIs are all either T1 Post contrast or T2. No meningioma included because of non-standardized imaging yet

% Creating Training and Testing Datasets
t1 = imageDatastore("T1 Post", 'IncludeSubfolders',true,'LabelSource','foldernames');
t2 = imageDatastore("T2", 'IncludeSubfolders',true,'LabelSource','foldernames');

% Training on only a subset of the training images (reducing training images)
trainingPercentage = 0.7;
[t1train, t1test] = splitEachLabel(t1, trainingPercentage, 'randomized');
[t2train, t2test] = splitEachLabel(t2, trainingPercentage, 'randomized');

% Image Preprocessing
T1Training_ds = augmentedImageDatastore([227 227],t1train,'ColorPreprocessing','gray2rgb');
T1Testing_ds = augmentedImageDatastore([227 227],t1test,'ColorPreprocessing','gray2rgb');
T2Training_ds = augmentedImageDatastore([227 227],t2train,'ColorPreprocessing','gray2rgb');
T2Testing_ds = augmentedImageDatastore([227 227],t2test,'ColorPreprocessing','gray2rgb');

% Initializing Network Layers
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(4);
layers(end) = classificationLayer;

% Setting Training Options
trainOpts1 = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'ValidationData',...
T1Training_ds,'ValidationFrequency',3,'Shuffle','every-epoch','MaxEpochs',6,'Plots','training-progress'); % T1 data

trainOpts2 = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'ValidationData',...
T2Training_ds,'ValidationFrequency',3,'Shuffle','every-epoch','MaxEpochs',6,'Plots','training-progress'); % T2 data

% Training the Networks
[T1Net,info] = trainNetwork(T1Training_ds,layers,trainOpts1);
[T2Net,info] = trainNetwork(T2Training_ds,layers,trainOpts2);


% Testing the Networks
Preds1 = classify(T1Net,T1Testing_ds);
truetest1 = t1test.Labels;
nnz(Preds1 == truetest1)/numel(Preds1)
confusionchart(truetest1,Preds1);

Preds2 = classify(T2Net,T2Testing_ds);
truetest2 = t2test.Labels;
nnz(Preds2 == truetest2)/numel(Preds2)
confusionchart(truetest2,Preds2);

%% T1 and T2 feature spaces

load('T1Net.mat');
load('T2Net.mat');

load('T1Testing_ds.mat');
load('T2Testing_ds.mat');

layer = 'fc';
T1NetfeaturesTest = activations(T1Net,T1Testing_ds,layer,'OutputAs','rows');
T2NetfeaturesTest = activations(T2Net,T2Testing_ds,layer,'OutputAs','rows');

% Reducing Dimensionality
[coeff1,score1] = pca(T1NetfeaturesTest);
[coeff2,score2] = pca(T2NetfeaturesTest);
T1Net_Acts = score1(:,1:2);
T2Net_Acts = score2(:,1:2);

% Assigning Tumor Clusters
T1Astro = T1Net_Acts(1:22,:);
T1Normal = T1Net_Acts(23:58,:);
T1OA = T1Net_Acts(59:66,:);
T1OD = T1Net_Acts(67:79,:);

T2Astro = T2Net_Acts(1:43,:);
T2Normal = T2Net_Acts(44:77,:);
T2OA = T2Net_Acts(78:122,:);
T2OD = T2Net_Acts(123:168,:);

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



