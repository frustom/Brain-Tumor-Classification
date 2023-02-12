%% ROC Curves
% Generating ROC curves using the test image activation nodes of the 
% Exp_Tumor_net (Meningioma Vs Glioma) network w/ 88.76% accuracy.

%% Generating Curves (Grouping Glioma and Meningioma activations into 1 category)

% Loading Previously Trained Network
load('Exp_Tumor_net.mat')

% Loading Testing Dataset
testds = imageDatastore('Testing','IncludeSubFolders',true,'LabelSource','foldernames');
Testing_ds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Extracting activations from FC (23rd) Layer
layer = 'fc';
Test_acts = activations(Exp_Tumor_net,Testing_ds,layer,'OutputAs','rows');

% Finding the Mean, Max, and Min of Each Category Node
Gnode = Test_acts(:,1); % 1:146
Gmax = max(Gnode);
Gmin = min(Gnode);
Gmean = mean(Gnode);

Hnode = Test_acts(:,2); % 147:205
Hmax = max(Hnode);
Hmin = min(Hnode);
Hmean = mean(Hnode);

Mnode = Test_acts(:,3); % 206:267
Mmax = max(Mnode);
Mmin = min(Mnode);
Mmean = mean(Mnode);

% Normalizing Activity Of Cancer Nodes
Gnorm = (Gnode - Gmin)/(Gmax - Gmin);
Mnorm = (Mnode - Mmin)/(Mmax - Mmin);
Cnorm = (Gnorm + Mnorm)/2;
Cmax = max(Cnorm);
Cmin = min(Cnorm);
Cmean = mean(Cnorm);
Cnorm = (Cnorm - Cmin)/(Cmax - Cmin);

% Normalizing Activity of Healthy Node
Hnorm = (Hnode - Hmin)/(Hmax - Hmin);

% Generating ROC Curve
counter = 0;
for alpha = 0.0001:0.0001:1
    counter = counter + 1;
    Hit_raw1 = (Cnorm(1:146) - alpha) > 0;
    Hit_raw2 = (Cnorm(206:207) - alpha) > 0;
    Hit_raw = [Hit_raw1;Hit_raw2];
    Hit_rate(counter) = sum(Hit_raw)/length(Hit_raw);
    FA_raw = (Cnorm(147:205) - alpha) > 0;
    FA_rate(counter) = sum(FA_raw)/length(FA_raw);
end
figure;
plot(FA_rate,Hit_rate,'r','LineWidth',2)
title('ExpCamoTumorNet ROC Curves') 
xlabel('False Alarm')
ylabel('Hit Rate')


%% Generating Curves (Keeping Glioma and Meningioma Activations Separate)

% Loading Previously Trained Network
load('Exp_Tumor_net.mat')

% Loading Testing Dataset
testds = imageDatastore('Testing','IncludeSubFolders',true,'LabelSource','foldernames');
Testing_ds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Extracting activations from FC (23rd) Layer
layer = 'fc';
Test_acts = activations(Exp_Tumor_net,Testing_ds,layer,'OutputAs','rows');

% Finding the Mean, Max, and Min of Each Category Node
Gnode = Test_acts(:,1); % 1:146
Gmax = max(Gnode);
Gmin = min(Gnode);
Gmean = mean(Gnode);

Mnode = Test_acts(:,3); % 206:267
Mmax = max(Mnode);
Mmin = min(Mnode);
Mmean = mean(Mnode);

% Normalizing Activity of Cancer Nodes
Gnorm = (Gnode - Gmin)/(Gmax - Gmin);
Mnorm = (Mnode - Mmin)/(Mmax - Mmin);

% Generating ROC Curve 1 (Hit = Glioma, FA = Meningioma)
counter = 0;
for alpha = 0.1:0.0001:1
    counter = counter + 1;
    Hit_raw =(Gnorm(1:146) - alpha) > 0;
    Hit_rate(counter) = sum(Hit_raw)/length(Hit_raw);
    FA_raw = (Gnorm(206:267) - alpha) > 0;
    FA_rate(counter) = sum(FA_raw)/length(FA_raw);
end
figure;
plot(FA_rate,Hit_rate,'blue','LineWidth',2)
title('ExpTumorNet Glioma vs Meningioma ROC Curve') 
xlabel('Meningioma False Alarm Rate')
ylabel('Glioma Hit Rate')

% Generating ROC Curve 2 (Hit = Meningioma, FA = Glioma)
counter = 0;
for alpha = 0.1:0.0001:1
    counter = counter + 1;
    Hit_raw =(Mnorm(206:267) - alpha) > 0;
    Hit_rate(counter) = sum(Hit_raw)/length(Hit_raw);
    FA_raw = (Mnorm(1:146) - alpha) > 0;
    FA_rate(counter) = sum(FA_raw)/length(FA_raw);
end
figure;
plot(FA_rate,Hit_rate,'g','LineWidth',1)
title('ExpTumorNet Meningioma vs Glioma ROC Curve') 
xlabel('Glioma False Alarm Rate')
ylabel('Meningioma Hit Rate')

%% Generating ROC curve (Removing Healthy Images)

% Loading Previously Trained Network
load('Exp_Tumor_net.mat')

% Loading Testing Dataset
testds = imageDatastore('Testing','IncludeSubFolders',true,'LabelSource','foldernames');
Testing_ds = augmentedImageDatastore([227 227],testds,'ColorPreprocessing','gray2rgb');

% Extracting activations from FC (23rd) Layer
layer = 'fc';
Test_acts = activations(Exp_Tumor_net,Testing_ds,layer,'OutputAs','rows');

GG_acts = Test_acts(1:146,1);
GM_acts = Test_acts(1:146,3);
MG_acts = Test_acts(206:267,1);
MM_acts = Test_acts(206:267,3);
Comb_acts = [GG_acts GM_acts;MG_acts MM_acts];

% Finding the Mean, Max, and Min of Each Category Node
Gnode = Comb_acts(:,1);
Gmax = max(Gnode);
Gmin = min(Gnode);
Gmean = mean(Gnode);

Mnode = Comb_acts(:,2);
Mmax = max(Mnode);
Mmin = min(Mnode);
Mmean = mean(Mnode);

% Normalizing Activity of Cancer Nodes
Gnorm = (Gnode - Gmin)/(Gmax - Gmin);
Mnorm = (Mnode - Mmin)/(Mmax - Mmin);

% Generating ROC Curve 1 (Hit = Glioma, FA = Meningioma)
counter = 0;
for alpha = 0.1:0.0001:1
    counter = counter + 1;
    Hit_raw =(Gnorm(1:146) - alpha) > 0;
    Hit_rate(counter) = sum(Hit_raw)/length(Hit_raw);
    FA_raw = (Gnorm(147:208) - alpha) > 0;
    FA_rate(counter) = sum(FA_raw)/length(FA_raw);
end
figure;
plot(FA_rate,Hit_rate)
title('ExpTumorNet Glioma vs Meningioma ROC Curve (w/o Healthy Activations)') 
xlabel('Meningioma False Alarm Rate')
ylabel('Glioma Hit Rate')

% Generating ROC Curve 2 (Hit = Meningioma, FA = Glioma)
counter = 0;
for alpha = 0.1:0.0001:1
    counter = counter + 1;
    Hit_raw =(Mnorm(147:208) - alpha) > 0;
    Hit_rate(counter) = sum(Hit_raw)/length(Hit_raw);
    FA_raw = (Mnorm(1:146) - alpha) > 0;
    FA_rate(counter) = sum(FA_raw)/length(FA_raw);
end
figure;
plot(FA_rate,Hit_rate,'g','LineWidth',2)
title('ExpTumorNet Meningioma vs Glioma ROC Curve (w/o Healthy Activations)') 
xlabel('Glioma False Alarm Rate')
ylabel('Meningioma Hit Rate')


% Hisotgram (Glioma is target tumor)
% hist(Gnorm(1:146))
% hold on
% hist(Gnorm(147:208))
% 
% figure;
% [N1,X1] = hist(Gnorm(1:146));
% Bh = bar(X1,N1,'facecolor','r')
% hold on
% [N2,X2] = hist(Gnorm(147:208));
% Bh1 = bar(X2,N2,'facecolor','b')
% hold off
% 
% figure;
% [N3,X3] = hist(Gnorm(1:146),7);
% Bh3 = bar(X3,N3,'facecolor','g')
% hold on
% [N4,X4] = hist(Gnorm(147:208),7);
% Bh4 = bar(X4,N4,'facecolor','y')
% 
% 
% figure;
% [N5,X5] = hist(Mnorm(1:146),10);
% Bh5 = bar(X5,N5,'facecolor','r');
% hold on
% [N6,X6] = hist(Mnorm(147:208));
% Bh6 = bar(X6,N6,'facecolor','b')
% hold off
% 
% figure;
% [N7,X7] = hist(Mnorm(1:146),7);
% Bh7 = bar(X7,N7,'facecolor','g')
% hold on
% [N8,X8] = hist(Mnorm(147:208),7);
% Bh8 = bar(X8,N8,'facecolor','y')














