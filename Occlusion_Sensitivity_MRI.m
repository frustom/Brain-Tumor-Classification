load('glioma_net.mat')
AstroG2_img1 = imread('AstroG2_test_1.jpg');
AstroG2_img2 = imread('AstroG2_test_2.jpg');
AstroG3_img1 = imread('AstroG3_test_1.jpg');
AstroG3_img2 = imread('AstroG3_test_2.jpg');
OAG2_img1 = imread('OAG2_test_1.png');
OAG2_img2 = imread('OAG2_test_2.jpg');
OAG3_img1 = imread('OAG3_test_1.jpg');
OAG3_img2 = imread('OAG3_test_2.jpg');
ODG2_img1 = imread('ODG2_test_1.png');
ODG2_img2 = imread('ODG2_test_2.jpg');
ODG3_img1 = imread('ODG3_test_1.jpg');
ODG3_img2 = imread('ODG3_test_2.jpg');

AstroG2_img1 = imresize(AstroG2_img1,[227 227]);
AstroG2_img2 = imresize(AstroG2_img2,[227 227]);
AstroG3_img1 = imresize(AstroG3_img1,[227 227]);
AstroG3_img2 = imresize(AstroG3_img2,[227 227]);
OAG2_img1 = imresize(OAG2_img1, [227 227]);
OAG2_img2 = imresize(OAG2_img2, [227 227]);
OAG3_img1 = imresize(OAG3_img1, [227 227]);
OAG3_img2 = imresize(OAG3_img2, [227 227]);
ODG2_img1 = imresize(ODG2_img1, [227 227]);
ODG2_img2 = imresize(ODG2_img2, [227 227]);
ODG3_img1 = imresize(ODG3_img1, [227 227]);
ODG3_img2 = imresize(ODG3_img2, [227 227]);

AstroG2_img1 = repmat(AstroG2_img1,[1,1,3]);
AstroG2_img2 = repmat(AstroG2_img2,[1,1,3]);
AstroG3_img1 = repmat(AstroG3_img1,[1,1,3]);
AstroG3_img2 = repmat(AstroG3_img2,[1,1,3]);
OAG2_img1 = repmat(OAG2_img1,[1,1,3]);
OAG2_img2 = repmat(OAG2_img2,[1,1,3]);
OAG3_img1 = repmat(OAG3_img1,[1,1,3]);
OAG3_img2 = repmat(OAG3_img2,[1,1,3]);
ODG2_img1 = repmat(ODG2_img1,[1,1,3]);
ODG2_img2 = repmat(ODG2_img2,[1,1,3]);
ODG3_img1 = repmat(ODG3_img1,[1,1,3]);
ODG3_img2 = repmat(ODG3_img2,[1,1,3]);

classes = glioma_net.Layers(end).Classes;

imshow(AstroG2_img1)
imshow(AstroG2_img2)
imshow(AstroG3_img1)
imshow(AstroG3_img2)
imshow(OAG2_img1)
imshow(OAG2_img2)
imshow(OAG3_img1)
imshow(OAG3_img2)
imshow(ODG2_img1)
imshow(ODG2_img2)
imshow(ODG3_img1)
imshow(ODG3_img2)

[Ypred1,scores1] = classify(glioma_net,AstroG2_img1); % Correct Label (57%)
[Ypred2,scores2] = classify(glioma_net,AstroG2_img2); % Correct Label (95%)
[Ypred3,scores3] = classify(glioma_net,AstroG3_img1); % Incorrect Label (41%)
[Ypred4,scores4] = classify(glioma_net,AstroG3_img2); % Correct Label (79%)
[Ypred5,scores5] = classify(glioma_net,OAG2_img1); % Correct Label (96%)
[Ypred6,scores6] = classify(glioma_net,OAG2_img2); % Correct Label (86%)
[Ypred7,scores7] = classify(glioma_net,OAG3_img1); % Correct Label (98%)
[Ypred8,scores8] = classify(glioma_net,OAG3_img2); % Correct Label (98%)
[Ypred9,scores9] = classify(glioma_net,ODG2_img1); % Correct Label (58%)
[Ypred10,scores10] = classify(glioma_net,ODG2_img2); % Correct Label (59%)
[Ypred11,scores11] = classify(glioma_net,ODG3_img1); % Correct Label (95%)
[Ypred12,scores12] = classify(glioma_net,ODG3_img2); % Correct Label (96%)


% Sensitivity Map for Incorrect Label (astrocytoma)
[~,topIdx3] = maxk(scores3, 3);
topScores3 = scores3(topIdx3);
topClasses3 = classes(topIdx3);
imshow(AstroG3_img1)
map3 = occlusionSensitivity(glioma_net,AstroG3_img1,Ypred3);
figure;
imshow(AstroG3_img1,'InitialMagnification', 150)
hold on
imagesc(map3,'AlphaData',0.5)
colormap jet
colorbar

% Sensitivity Map for Correct Label - astrocytoma (Low Confidence)
[~,topIdx1] = maxk(scores1, 3);
topScores1 = scores1(topIdx1);
topClasses1 = classes(topIdx1);
imshow(AstroG2_img1)
map1 = occlusionSensitivity(glioma_net,AstroG2_img1,Ypred1);
figure;
imshow(AstroG2_img1,'InitialMagnification', 150)
hold on
imagesc(map1,'AlphaData',0.5)
colormap jet
colorbar

% Sensitivity Map for Correct Label - astrocytoma (high confidence)
[~,topIdx2] = maxk(scores2, 3);
topScores2 = scores2(topIdx2);
topClasses2 = classes(topIdx2);
imshow(AstroG2_img2)
map2 = occlusionSensitivity(glioma_net,AstroG2_img2,Ypred2);
figure;
imshow(AstroG2_img2,'InitialMagnification', 150)
hold on
imagesc(map2,'AlphaData',0.5)
colormap jet
colorbar

% Sensitivity Map for Correct Label - Oligoastrocytoma (high confidence)
[~,topIdx7] = maxk(scores7, 3);
topScores7 = scores7(topIdx7);
topClasses7 = classes(topIdx7);
imshow(OAG3_img1)
map7 = occlusionSensitivity(glioma_net,OAG3_img1,Ypred7);
figure;
imshow(OAG3_img1,'InitialMagnification', 150)
hold on
imagesc(map7,'AlphaData',0.5)
colormap jet
colorbar

% Sensitivity Map for Correct Label - Oligodendroglioma (high confidence)
[~,topIdx12] = maxk(scores12, 3);
topScores12 = scores12(topIdx12);
topClasses12 = classes(topIdx12);
imshow(ODG3_img2)
map12 = occlusionSensitivity(glioma_net,ODG3_img2,Ypred12);
figure;
imshow(ODG3_img2,'InitialMagnification', 150)
hold on
imagesc(map12,'AlphaData',0.5)
colormap jet
colorbar

% Sensitivity Map for Correct Label - Oligodendroglioma (low confidence)
[~,topIdx10] = maxk(scores10, 3);
topScores10 = scores10(topIdx10);
topClasses10 = classes(topIdx10);
imshow(ODG2_img2)
map10 = occlusionSensitivity(glioma_net,ODG2_img2,Ypred10);
figure;
imshow(ODG2_img2,'InitialMagnification', 150)
hold on
imagesc(map10,'AlphaData',0.5)
colormap jet
colorbar














