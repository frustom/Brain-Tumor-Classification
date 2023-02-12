%% Glioma Classifier 2 PCA
% Looking at the feature space of glioma_net after removing the OA category

%% Dimensionality Reduction and Feature Mapping

% Loading previously trained networks
load('glioma_net2.mat');

% Loading Augmented Testing Image Datastores
load('Glioma_Testds.mat');

% Extracting activations from FC (23rd) Layer
layer = 'fc';
glioma_net2_Test = activations(glioma_net2,Glioma_Testds,layer,'OutputAs','rows');

% Reducing Dimensionality
[coeff, score] = pca(glioma_net2_Test);
glioma_acts = score(:,1:2);

% Mapping Glioma Test Image Clusters
figure;
hold on
plot([glioma_acts(1:75,1)],[glioma_acts(1:75,2)],'bo')
plot([glioma_acts(76:288,1)],[glioma_acts(76:288,2)],'go')
xlim([-10 10])
ylim([-10 10])
title('GliomaNet Test Activations')
legend('Astrocytoma','Oligodendroglioma','Location','Best')

%% After Transfer Learning

% Extracting activations from FC (23rd) Layer
layer = 'fc';
expglioma_net2_Test = activations(ExpCamo_Glioma_net,Glioma_Testds,layer,'OutputAs','rows');

% Reducing Dimensionality
[coeff, score] = pca(expglioma_net2_Test);
glioma_acts = score(:,1:2);

% Mapping Glioma Test Image Clusters
figure;
hold on
plot([glioma_acts(1:75,1)],[glioma_acts(1:75,2)],'bo')
plot([glioma_acts(76:288,1)],[glioma_acts(76:288,2)],'go')
xlim([-10 10])
ylim([-10 10])
title('ExpCamoGliomaNet2 Test Activations')
legend('Astrocytoma','Oligodendroglioma','Location','Best','FontSize',14)
