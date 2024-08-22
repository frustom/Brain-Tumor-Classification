%% Image Sensitivity for Revised Dataset Part 1
% Trying occlusionSensitivity and gradCAM iamge sensitivity functions on new image dataset for MS2 revision

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

%% gradCAM (fullyConnected) T1Net
figure
subplot(2,2,1)
scoreMap = gradCAM(T1Net,T1astro,'Astrocytoma',ReductionLayer="fc");
imshow(T1astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = gradCAM(T1Net,T1OA,'Oligoastrocytoma',ReductionLayer="fc");
imshow(T1OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = gradCAM(T1Net,T1OD,'Oligodendroglioma',ReductionLayer="fc");
imshow(T1OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = gradCAM(T1Net,T1normal,'Normal',ReductionLayer="fc");
imshow(T1normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% gradCAM (softmax) T1Net
figure
subplot(2,2,1)
activations(T1Net,T1astro,'prob','OutputAs','rows');
pred = classify(T1Net, T1astro);
scoreMap = gradCAM(T1Net, T1astro, pred);
imshow(T1astro)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet
colorbar

subplot(2,2,2)
activations(T1Net,T1OA,'prob','OutputAs','rows');
pred = classify(T1Net, T1OA);
scoreMap = gradCAM(T1Net, T1OA, pred);
imshow(T1OA)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet
colorbar

subplot(2,2,3)
activations(T1Net,T1OD,'prob','OutputAs','rows');
pred = classify(T1Net, T1OD);
scoreMap = gradCAM(T1Net, T1OD, pred);
imshow(T1OD)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet
colorbar

subplot(2,2,4)
activations(T1Net,T1normal,'prob','OutputAs','rows');
pred = classify(T1Net, T1normal);
scoreMap = gradCAM(T1Net, T1normal, pred);
imshow(T1normal)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet
colorbar

%% gradCAM (fullyConnected) T2Net
figure
subplot(2,2,1)
scoreMap = gradCAM(T2Net,T2astro,'Astrocytoma',ReductionLayer="fc");
imshow(T2astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = gradCAM(T2Net,T2OA,'Oligoastrocytoma',ReductionLayer="fc");
imshow(T2OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = gradCAM(T2Net,T2OD,'Oligodendroglioma',ReductionLayer="fc");
imshow(T2OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = gradCAM(T2Net,T2normal,'Normal',ReductionLayer="fc");
imshow(T2normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% gradCAM (softmax) T2Net
figure
subplot(2,2,1)
scoreMap = gradCAM(T2Net,T2astro,'Astrocytoma',ReductionLayer="prob");
imshow(T2astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = gradCAM(T2Net,T2OA,'Oligoastrocytoma',ReductionLayer="prob");
imshow(T2OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = gradCAM(T2Net,T2OD,'Oligodendroglioma',ReductionLayer="prob");
imshow(T2OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = gradCAM(T2Net,T2normal,'Normal',ReductionLayer="prob");
imshow(T2normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% gradCAM (fullyConnected) ExpT1Net
figure
subplot(2,2,1)
scoreMap = gradCAM(ExpT1Net,T1astro,'Astrocytoma',ReductionLayer="fc");
imshow(T1astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = gradCAM(ExpT1Net,T1OA,'Oligoastrocytoma',ReductionLayer="fc");
imshow(T1OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = gradCAM(ExpT1Net,T1OD,'Oligodendroglioma',ReductionLayer="fc");
imshow(T1OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = gradCAM(ExpT1Net,T1normal,'Normal',ReductionLayer="fc");
imshow(T1normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% gradCAM (softmax) ExpT1Net
figure
subplot(2,2,1)
scoreMap = gradCAM(ExpT1Net,T1astro,'Astrocytoma',ReductionLayer="prob");
imshow(T1astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = gradCAM(ExpT1Net,T1OA,'Oligoastrocytoma',ReductionLayer="prob");
imshow(T1OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = gradCAM(ExpT1Net,T1OD,'Oligodendroglioma',ReductionLayer="prob");
imshow(T1OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = gradCAM(ExpT1Net,T1normal,'Normal',ReductionLayer="prob");
imshow(T1normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% gradCAM (fullyConnected) ExpT2Net
figure
subplot(2,2,1)
scoreMap = gradCAM(ExpT2Net,T2astro,'Astrocytoma',ReductionLayer="fc");
imshow(T2astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = gradCAM(ExpT2Net,T2OA,'Oligoastrocytoma',ReductionLayer="fc");
imshow(T2OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = gradCAM(ExpT2Net,T2OD,'Oligodendroglioma',ReductionLayer="fc");
imshow(T2OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = gradCAM(ExpT2Net,T2normal,'Normal',ReductionLayer="fc");
imshow(T2normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

%% gradCAM (softmax) ExpT2Net
figure
subplot(2,2,1)
scoreMap = gradCAM(ExpT2Net,T2astro,'Astrocytoma',ReductionLayer="prob");
imshow(T2astro)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,2)
scoreMap = gradCAM(ExpT2Net,T2OA,'Oligoastrocytoma',ReductionLayer="prob");
imshow(T2OA)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,3)
scoreMap = gradCAM(ExpT2Net,T2OD,'Oligodendroglioma',ReductionLayer="prob");
imshow(T2OD)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar

subplot(2,2,4)
scoreMap = gradCAM(ExpT2Net,T2normal,'Normal',ReductionLayer="prob");
imshow(T2normal)
hold on
imagesc(scoreMap,AlphaData=0.5)
colormap jet
colorbar