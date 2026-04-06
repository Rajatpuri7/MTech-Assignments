%% ============================================================
% AUTONOMOUS VEHICLE STEERING (CNN)
% ============================================================

clc; clear; close all;

%% ============================================================
% STEP 1: SET DIRECTORY
% ============================================================
cd('C:\Users\lnovo\Desktop\AI\Assignments\Steering Angle')
disp('Step 1: Working directory set');

%% ============================================================
% STEP 2: LOAD DATA 
% ============================================================
disp('Step 2: Loading dataset...');

data1 = readtable('self_driving_car_dataset_jungle/driving_log.csv', ...
    'Delimiter', ',', 'ReadVariableNames', false, ...
    'Format','%s%s%s%f%f%f%f');

data2 = readtable('self_driving_car_dataset_make/driving_log.csv', ...
    'Delimiter', ',', 'ReadVariableNames', false, ...
    'Format','%s%s%s%f%f%f%f');

%% ============================================================
% STEP 3: EXTRACT DATA
% ============================================================
disp('Step 3: Extracting data');

img1 = string(data1.Var1);
ang1 = data1.Var4;

img2 = string(data2.Var1);
ang2 = data2.Var4;

imagePaths = [img1; img2];
angles = [ang1; ang2];

%% ============================================================
% STEP 4: FIX PATHS
% ============================================================
disp('Step 4: Fixing paths');

imagePaths = imagePaths(:);

for i = 1:length(imagePaths)
    [~, name, ext] = fileparts(char(imagePaths(i)));
    
    if contains(imagePaths(i), 'jungle')
        imagePaths(i) = fullfile('self_driving_car_dataset_jungle','IMG',[name ext]);
    else
        imagePaths(i) = fullfile('self_driving_car_dataset_make','IMG',[name ext]);
    end
end

%% ============================================================
% STEP 5: MULTI-CAMERA AUGMENTATION
% ============================================================
disp('Step 5: Using 3 cameras');

correction = 0.2;

allImages = [];
allAngles = [];

for i = 1:length(imagePaths)
    
    % center
    allImages = [allImages; imagePaths(i)];
    allAngles = [allAngles; angles(i)];
    
    % left
    leftPath = strrep(imagePaths(i), 'center', 'left');
    allImages = [allImages; leftPath];
    allAngles = [allAngles; angles(i) + correction];
    
    % right
    rightPath = strrep(imagePaths(i), 'center', 'right');
    allImages = [allImages; rightPath];
    allAngles = [allAngles; angles(i) - correction];
    
end

imagePaths = allImages;
angles = allAngles;

disp(['Total samples after augmentation: ', num2str(length(angles))]);

%% ============================================================
% STEP 6: DATASTORE
% ============================================================
imds = imageDatastore(imagePaths);

%% ============================================================
% STEP 7: CHECK IMAGE
% ============================================================
img = readimage(imds,22000);
figure; imshow(img);
title('Raw Input Image');

%% ============================================================
% STEP 8: REMOVE BIAS
% ============================================================
disp('Step 8: Removing bias');

idx = abs(angles) > 0.05 | rand(size(angles)) < 0.05;

imds = subset(imds, find(idx));
angles = angles(idx);

%% ============================================================
% STEP 9: COMBINE
% ============================================================
ds = combine(imds, arrayDatastore(angles));

%% ============================================================
% STEP 10: PREPROCESS
% ============================================================
ds = transform(ds, @preprocessData);

sample = read(ds);
figure; imshow(sample{1}, []);
title(['Processed Image | Angle = ', num2str(sample{2})]);

%% ============================================================
% STEP 11: SPLIT
% ============================================================
numData = numel(angles);
idx = randperm(numData);

trainIdx = idx(1:round(0.8*numData));
testIdx = idx(round(0.8*numData)+1:end);

dsTrain = subset(ds, trainIdx);
dsTest = subset(ds, testIdx);

%% ============================================================
% STEP 12: CNN MODEL
% ============================================================
layers = [
    imageInputLayer([66 200 3])

    convolution2dLayer(5,24,'Stride',2)
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(5,36,'Stride',2)
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(5,48,'Stride',2)
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,64)
    reluLayer

    fullyConnectedLayer(100)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(50)
    reluLayer

    fullyConnectedLayer(1)
    regressionLayer
];

%% ============================================================
% STEP 13: TRAIN
% ============================================================
disp('Training network...');

options = trainingOptions('adam', ...
    'MaxEpochs', 8, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-4, ...
    'Plots','training-progress');

net = trainNetwork(dsTrain, layers, options);

%% ============================================================
% STEP 14: PREDICT
% ============================================================
YPred = predict(net, dsTest);

%% ============================================================
% STEP 15: TRUE VALUES
% ============================================================
dataTest = readall(dsTest);
YTest = cell2mat(dataTest(:,2));

%% ============================================================
% STEP 16: RESULTS
% ============================================================
figure;
plot(YTest,'b'); hold on;
plot(YPred,'r');
legend('Actual','Predicted');
title('Steering Prediction');

rmse = sqrt(mean((YPred - YTest).^2));
disp(['RMSE = ', num2str(rmse)]);

disp('Training Complete!');

%% ============================================================
% PREPROCESS FUNCTION
% ============================================================
function dataOut = preprocessData(data)
    img = data{1};
    angle = data{2};

    img = img(60:end-25,:,:); % crop
    img = imresize(img, [66 200]);
    img = img / 255;

    dataOut = {img, angle};
end