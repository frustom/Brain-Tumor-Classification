%% Transfer Learning Glioma Classifier
% Performing transfer learning of an animal camouflage detection network (both transfer learning paradigms)
% onto a glioma classifier to compare accuracy and feature space. The exp_camo_net and exp_clear_net loaded
% here is the same network used in earlier feature space generation.

%% Building the Network

% Creating Datastores for training testing images
trainds = imageDatastore('Combined Train','IncludeSubFolders',true,'LabelSource','foldernames');
testds = imageDatastore('Combined Test','IncludeSubFolders',true,'LabelSource','foldernames');

% Image Preprocessing
Glioma_Trainds = augmentedImageDatastore([227 227],trainds,'ColorPreprocessing','gray2rgb');
Glioma_Testds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Loading in trained networks for comparison
load('exp_camo_net.mat');
load('exp_clear_net.mat');

% Extracting Previously Trained Camo Layers
layers = exp_camo_net.Layers;
layers(end-2) = fullyConnectedLayer(3);
layers(end) = classificationLayer;

% Extracting Previously Trained Clear Layers
layers2 = exp_clear_net.Layers;
layers2(end-2) = fullyConnectedLayer(3);
layers2(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.1,'ValidationData',Glioma_Trainds,'ValidationFrequency',...
5,'Shuffle','once','MaxEpochs',50,'Plots','training-progress');


% Training Networks
[ExpCamo_Glioma_net,info] = trainNetwork(Glioma_Trainds,layers,trainOpts);
[ExpClear_Glioma_net,info] = trainNetwork(Glioma_Trainds,layers2,trainOpts);

% Testing Networks
preds = classify(ExpCamo_Glioma_net,Glioma_Testds); % exp_camo_net TL
truetest = testds.Labels;
accuracy = nnz(preds == truetest)/numel(preds)
confusionchart(truetest,preds);
% G2 = 0.9368 accuracy
% G3 = 0.9280 accuracy
% Combined = 0.9327

preds2 = classify(ExpClear_Glioma_net,Glioma_Testds); % exp_clear_net TL
truetest2 = testds.Labels;
accuracy2 = nnz(preds2 == truetest2)/numel(preds2)
figure;
confusionchart(truetest2,preds2);
% G2 = 0.8885% 
% G3 = 0.9364%
% combined = 0.9109

% title(['ExpCamoGliomaNet Grade 2 Test Activations (' num2str(93.68) '% Accuracy)'])


