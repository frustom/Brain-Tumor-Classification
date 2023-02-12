%% Transfer Learning Glioma Classifier
% Performing transfer learning of an animal camouflage detection network onto a glioma classifier to compare accuracy and feature space. 
% The exp_camo_net loaded here is the same network used in earlier feature space generation.

%% Building the Network

% Creating Datastores for training testing images
trainds = imageDatastore('Train','IncludeSubFolders',true,'LabelSource','foldernames');
testds = imageDatastore('Test','IncludeSubFolders',true,'LabelSource','foldernames');

% Image Preprocessing
Glioma_Trainds = augmentedImageDatastore([227 227],trainds,'ColorPreprocessing','gray2rgb');
Glioma_Testds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Loading in trained networks for comparison
load('exp_camo_net.mat');

% Extracting Previously Trained Camo Layers
layers = exp_camo_net.Layers;
layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.1,'ValidationData',Glioma_Trainds,'ValidationFrequency',...
5,'Shuffle','once','MaxEpochs',25,'Plots','training-progress');

% Training Networks
[ExpCamo_Glioma_net,info] = trainNetwork(Glioma_Trainds,layers,trainOpts);

% Testing Networks
preds = classify(ExpCamo_Glioma_net,Glioma_Testds); % exp_camo_net TL
truetest = testds.Labels;
accuracy = nnz(preds == truetest)/numel(preds)
confusionchart(truetest,preds);

% 98.61 w/o OA (after camo transfer learning)
