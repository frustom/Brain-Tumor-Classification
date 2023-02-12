%% Glioma Classifier (w/o OA)
% This network is trained and tested on a blend of Grade 2 and Grade 3 gliomas (astrocytomas and oligodendrogliomas)
% The Grade 2 and Grade 3 glioma datasets are kept separate in a different folder for future use in feature space analysis on glioma_net (trained here)
% *** modified version of glioma_classifier.m without OA)  ***

%% Building the Network

% Creating Datastores for training testing images
trainds = imageDatastore('Train','IncludeSubFolders',true,'LabelSource','foldernames');
testds = imageDatastore('Test','IncludeSubFolders',true,'LabelSource','foldernames');

% Image Preprocessing
Glioma_Trainds = augmentedImageDatastore([227 227],trainds,'ColorPreprocessing','gray2rgb');
Glioma_Testds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Modifying Pretrained Network
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.1,'ValidationData',Glioma_Trainds,'ValidationFrequency',...
5,'Shuffle','once','MaxEpochs',10,'Plots','training-progress');

% Training the Network
[glioma_net2, info] = trainNetwork(Glioma_Trainds,layers,trainOpts);

% Testing Network
preds = classify(glioma_net2,Glioma_Testds);
truetest = testds.Labels;
accuracy = nnz(preds == truetest)/numel(preds)
confusionchart(truetest,preds);

% 97.22 accuracy w/o oligoastrocytomas



