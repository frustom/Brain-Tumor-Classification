%% Image Sensitivity for Revised Dataset Part 2
% Trying occlusionSensitivity and gradCAM iamge sensitivity functions on new image dataset for MS2 revision. OcclusionSnesitivity found in this code

% Loading networks
load("T1Net.mat");
load("T2Net.mat");
load("ExpT1Net.mat");
load("ExpT2Net.mat");

% Reading all images
T1astro = imread("T1 Post Astro.jpg");
T1normal = imread("T1 Post Normal.jpg");
T1OA = imread("T1 Post OA.jpg");
T1OD = imread("T1 Post OD.jpg");
T2astro = imread("T2 Astro.jpg");
T2normal = imread("T2 Normal.jpg");
T2OA = imread("T2 OA.jpg");
T2OD = imread("T2 OD.jpg");

% Resizing
T1astro = imresize(T1astro,[227,227]);
T1astro = repmat(T1astro, [1,1,3]);
T1normal = imresize(T1normal,[227,227]);
%T1normal = repmat(T1normal, [1,1,3]);
T1OA = imresize(T1OA,[227,227]);
T1OA = repmat(T1OA, [1,1,3]);
T1OD = imresize(T1OD,[227,227]);
T1OD = repmat(T1OD, [1,1,3]);

T2astro = imresize(T2astro,[227,227]);
T2astro = repmat(T2astro, [1,1,3]);
T2normal = imresize(T2normal,[227,227]);
%T2normal = repmat(T2normal, [1,1,3]);
T2OA = imresize(T2OA,[227,227]);
T2OA = repmat(T2OA, [1,1,3]);
T2OD = imresize(T2OD,[227,227]);
T2OD = repmat(T2OD, [1,1,3]);

%% occlusionSensitivity T1Net
figure
subplot(2,2,1)
scoreMap = occlusionSensitivity(T1Net,T1astro,'Astrocytoma');
imshow(T1astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = occlusionSensitivity(T1Net,T1OA,'Oligoastrocytoma');
imshow(T1OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = occlusionSensitivity(T1Net,T1OD,'Oligodendroglioma');
imshow(T1OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = occlusionSensitivity(T1Net,T1normal,'Normal');
imshow(T1normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% occlusionSensitivity T2Net
figure
subplot(2,2,1)
scoreMap = occlusionSensitivity(T2Net,T2astro,'Astrocytoma');
imshow(T2astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = occlusionSensitivity(T2Net,T2OA,'Oligoastrocytoma');
imshow(T2OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = occlusionSensitivity(T2Net,T2OD,'Oligodendroglioma');
imshow(T2OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = occlusionSensitivity(T2Net,T2normal,'Normal');
imshow(T2normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% occlusionSensitivity ExpT1Net
figure
subplot(2,2,1)
scoreMap = occlusionSensitivity(ExpT1Net,T1astro,'Astrocytoma');
imshow(T1astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = occlusionSensitivity(ExpT1Net,T1OA,'Oligoastrocytoma');
imshow(T1OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = occlusionSensitivity(ExpT1Net,T1OD,'Oligodendroglioma');
imshow(T1OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = occlusionSensitivity(ExpT1Net,T1normal,'Normal');
imshow(T1normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% occlusionSensitivity ExpT2Net
figure
subplot(2,2,1)
scoreMap = occlusionSensitivity(ExpT2Net,T2astro,'Astrocytoma');
imshow(T2astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = occlusionSensitivity(ExpT2Net,T2OA,'Oligoastrocytoma');
imshow(T2OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = occlusionSensitivity(ExpT2Net,T2OD,'Oligodendroglioma');
imshow(T2OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = occlusionSensitivity(ExpT2Net,T2normal,'Normal');
imshow(T2normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

 