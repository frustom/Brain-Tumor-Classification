%% Camouflage Classifier 8
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
% In this trial, a network is trained and tested on clear.
%% Building the Network

% Creating/Labeling DS for training & testing
Imgs = imageDatastore('New Clear','IncludeSubFolders',true,'LabelSource','foldernames');

% Splitting into training and testing
[trainImgs,testImgs] = splitEachLabel(Imgs,0.6,'randomized');

% Image Preprocessing
ClearTrainds = augmentedImageDatastore([227 227],trainImgs,'ColorPreprocessing','gray2rgb');
ClearTestds = augmentedImageDatastore([227 227],testImgs,'ColorPreprocessing','gray2rgb');

% Modifying the Network
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(15);
layers(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.05,'ValidationData',ClearTrainds,'ValidationFrequency',15,'Shuffle',...
'every-epoch','MaxEpochs',20,'Plots','training-progress');

% Training the network
[clear_net2, info] = trainNetwork(ClearTrainds,layers,trainOpts);

% Making predictions
preds = classify(clear_net2,ClearTestds);
truetest = testImgs.Labels;
nnz(preds == truetest)/numel(preds)
confusionchart(truetest,preds);

% Testing on Training Dataset (for MS)
preds2 = classify(clear_net2,ClearTrainds);
truetest2 = trainImgs.Labels;
nnz(preds2 == truetest2)/numel(preds2)
confusionchart(truetest2,preds2);

