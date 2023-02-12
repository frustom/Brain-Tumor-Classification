%% Glioma Classifier
% This network is trained and tested on a blend of Grade 2 and Grade 3 gliomas (astrocytomas, oligoastrocytomas, oligodendrogliomas)
% The Grade 2 and Grade 3 glioma datasets are kept separate in a different folder for future use in feature space analysis on glioma_net (trained here)

%% Building the Network

% Creating Datastores for training testing images
trainds = imageDatastore('Combined Train','IncludeSubFolders',true,'LabelSource','foldernames');
testds = imageDatastore('Combined Test','IncludeSubFolders',true,'LabelSource','foldernames');

% Image Preprocessing
Glioma_Trainds = augmentedImageDatastore([227 227],trainds,'ColorPreprocessing','gray2rgb');
Glioma_Testds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Modifying Pretrained Network
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(3);
layers(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.1,'ValidationData',Glioma_Trainds,'ValidationFrequency',...
5,'Shuffle','once','MaxEpochs',50,'Plots','training-progress');

% Training the Network
[glioma_net, info] = trainNetwork(Glioma_Trainds,layers,trainOpts);

% Testing Network
preds = classify(glioma_net,Glioma_Testds);
truetest = testds.Labels;
accuracy = nnz(preds == truetest)/numel(preds)
confusionchart(truetest,preds);

%% Creating and Preprocessing Specific Datastores
% Creating necessary datastores for future PCA and Feature Space Analysis
g2train = imageDatastore('G2 Train','IncludeSubFolders',true,'LabelSource','foldernames');
g2test = imageDatastore('G2 Test','IncludeSubFolders',true,'LabelSource','foldernames');
g3train = imageDatastore('G3 Train','IncludeSubFolders',true,'LabelSource','foldernames');
g3test = imageDatastore('G3 Test','IncludeSubFolders',true,'LabelSource','foldernames');

G2Glioma_Trainds = augmentedImageDatastore([227 227],g2train,'ColorPreprocessing','gray2rgb');
G2Glioma_Testds = augmentedImageDatastore([227 227],g2test,'ColorPreprocessing','gray2rgb');
G3Glioma_Trainds = augmentedImageDatastore([227 227],g3train,'ColorPreprocessing','gray2rgb');
G3Glioma_Testds = augmentedImageDatastore([227 227],g3test,'ColorPreprocessing','gray2rgb');


preds = classify(glioma_net,Glioma_Testds);
truetest = testds.Labels;
accuracy = nnz(preds == truetest)/numel(preds)
confusionchart(truetest,preds);


G2_acc = 0.9280
G3_acc = 0.9294


