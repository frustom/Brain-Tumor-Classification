%% Camouflage Classifier 9
% Training a CNN to recognize animals in camouflaged backgrounds for future
% use in transfer learning for early grade tumor detection.
% All images used for training are courtesy of:
% Anabranch Network for Camouflaged Object Segmentation
%    @article{ltnghia-CVIU2019,
%      author    = {Trung-Nghia Le and Tam V. Nguyen and Zhongliang Nie and Minh-Triet Tran and Akihiro Sugimoto,
%      journal   = {Computer Vision and Image Understanding Journal}, 
%      title     = {Anabranch Network for Camouflaged Object Segmentation}, 
%      year      = {2019}, 
%      volume    = {}, 
%      number    = {}, 
%      pages     = {-}, 
%
% In this trial, a network is trained on clear and tested on camo
%% Building the Network

% Creating/Labeling DS for training & testing
CleartrainImgs = imageDatastore('New Clear','IncludeSubFolders',true,'LabelSource','foldernames');
CamotestImgs = imageDatastore('New Camo','IncludeSubFolders',true,'LabelSource','foldernames');

% Split camo images to test on smaller portion
[testImgs1,CamotestImgs2] = splitEachLabel(CamotestImgs,0.7,'randomized');

% Image Preprocessing
trainds = augmentedImageDatastore([227 227],CleartrainImgs,'ColorPreprocessing','gray2rgb');
testds = augmentedImageDatastore([227 227],CamotestImgs2,'ColorPreprocessing','gray2rgb');

% Modifying the Network
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(15);
layers(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.05,'ValidationData',trainds,'ValidationFrequency',15,'Shuffle',...
'every-epoch','MaxEpochs',20,'Plots','training-progress');

% Training the network
[camo_net, info] = trainNetwork(trainds,layers,trainOpts);

% Making predictions
preds = classify(camo_net,testds);
truetest = CamotestImgs2.Labels;
nnz(preds == truetest)/numel(preds)
confusionchart(truetest,preds);
