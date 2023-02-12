%% ROC Curves
% Generating ROC curves using the test image activation nodes of the 
% ExpT2 (T1 -> T2) network w/ 94.48% accuracy.

%% Generating Curves

% Loading Previously Trained Network
load('ExpT2_net.mat')

% Loading Testing Dataset
testds = imageDatastore('T2 Testing','IncludeSubFolders',true,'LabelSource','foldernames');
T2_Testing_ds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Extracting activations from FC (23rd) Layer
layer = 'fc';
T2_acts = activations(ExpT2_net,T2_Testing_ds,layer,'OutputAs','rows');

% Finding the Mean, Max, and Min of Each Category Node
Anode = T2_acts(:,1); % 1:75
Amax = max(Anode);
Amin = min(Anode);
Amean = mean(Anode);

Hnode = T2_acts(:,2); % 76:95
Hmax = max(Hnode);
Hmin = min(Hnode);
Hmean = mean(Hnode);

OAnode = T2_acts(:,3); % 96:312
OAmax = max(OAnode);
OAmin = min(OAnode);
OAmean = mean(OAnode);

ODnode = T2_acts(:,4); % 313:525
ODmax = max(ODnode);
ODmin = min(ODnode);
ODmean = mean(ODnode);

% Normalizing Activity of Cancer Nodes
Anorm = (Anode - Amin)/(Amax - Amin);
OAnorm = (OAnode - OAmin)/(OAmax - OAmin);
ODnorm = (ODnode - ODmin)/(ODmax - ODmin);
Cnorm = (Anorm + OAnorm + ODnorm)/3;
Cmax = max(Cnorm);
Cmin = min(Cnorm);
Cmean = mean(Cnorm);
Cnorm = (Cnorm - Cmin)/(Cmax - Cmin);

% Normalizing Activity of Healthy Node
Hnorm = (Hnode - Hmin)/(Hmax - Hmin);

% Generating ROC Curve
counter = 0;
for alpha = 0.1:0.1:1
    counter = counter + 1;
    Hit_raw =(ODnorm(313:525) - alpha) > 0;
    Hit_rate(counter) = sum(Hit_raw)/length(Hit_raw);
    FA_raw = (ODnorm(76:95) - alpha) > 0;
    FA_rate(counter) = sum(FA_raw)/length(FA_raw);
end
plot(FA_rate,Hit_rate)


counter = 0;
for alpha = 0.0001:0.0001:1
    counter = counter + 1;
    Hit_raw1 =(Cnorm(1:75) - alpha) > 0;
    Hit_raw2 = (Cnorm(96:525) - alpha) > 0;
    Hit_raw = [Hit_raw1;Hit_raw2];
    Hit_rate(counter) = sum(Hit_raw)/length(Hit_raw);
    FA_raw = (Cnorm(76:95) - alpha) > 0;
    FA_rate(counter) = sum(FA_raw)/length(FA_raw);
end
plot(FA_rate,Hit_rate)
title('ExpT2 Glioma vs Healthy ROC Curve')
xlabel('Healthy False Alarm Rate')
ylabel('Glioma Hit Rate')



















