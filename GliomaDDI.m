%% Visualization of Glioma categories (Pedram Dataset)

% T1 Post Contrast Visualization
load('T1Net.mat')

% Visualize Fully Connected Layer
layer = 23;
name = T1Net.Layers(layer).Name;

channels = [1 4 3 2];
T1Net.Layers(end).Classes(channels)

I = deepDreamImage(T1Net,name,channels,'Verbose',false,'NumIterations',150,'PyramidLevels',3);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = T1Net.Layers(layer).Name;
title(['Layer ',name,' Features'])


%% T2 Visualization

% T1 Post Contrast Visualization
load('T2Net.mat')

% Visualize Fully Connected Layer
layer = 23;
name = T2Net.Layers(layer).Name;

channels = [1 4 3 2];
T2Net.Layers(end).Classes(channels)

I = deepDreamImage(T2Net,name,channels,'Verbose',false,'NumIterations',150,'PyramidLevels',3);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = T2Net.Layers(layer).Name;
title(['Layer ',name,' Features'])

%% ExpT1 and ExpT2Net Visualization

load('ExpT1Net.mat')
load('ExpT2Net.mat')

layer = 23;
name = ExpT1Net.Layers(layer).Name;

channels = [1 4 3 2];
ExpT1Net.Layers(end).Classes(channels)

I = deepDreamImage(ExpT1Net,name,channels,'Verbose',false,'NumIterations',150,'PyramidLevels',3);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = ExpT1Net.Layers(layer).Name;
title(['Layer ',name,' Features'])


% ExpT2
layer = 23;
name = ExpT2Net.Layers(layer).Name;

channels = [1 4 3 2];
ExpT2Net.Layers(end).Classes(channels)

I = deepDreamImage(ExpT2Net,name,channels,'Verbose',false,'NumIterations',150,'PyramidLevels',3);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = ExpT2Net.Layers(layer).Name;
title(['Layer ',name,' Features'])


















