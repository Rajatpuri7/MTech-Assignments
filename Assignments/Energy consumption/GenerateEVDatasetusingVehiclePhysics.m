clc
clear
close all

disp('===============================================')
disp('EV ENERGY CONSUMPTION PREDICTION DEMO')
disp('Physics + Machine Learning')
disp('===============================================')

%% =====================================================
% STEP 1: DATASET GENERATION USING VEHICLE PHYSICS
% =====================================================

N = 300;

speed = 20 + 100*rand(N,1);        % km/h
acceleration = -1 + 2*rand(N,1);   % m/s^2
roadGrade = -4 + 8*rand(N,1);      % degrees
temperature = 15 + 20*rand(N,1);   % degC

%% Vehicle parameters

m = 1600;
Cr = 0.012;
rho = 1.225;
Cd = 0.29;
A = 2.3;
g = 9.81;

%% Convert speed

v = speed/3.6;

%% Forces

F_roll = Cr*m*g;

F_aero = 0.5*rho*Cd*A.*v.^2;

F_grade = m*g.*sin(deg2rad(roadGrade));

F_acc = m.*acceleration;

F_total = F_roll + F_aero + F_grade + F_acc;

%% Traction power

P = F_total .* v;

%% Energy consumption (Wh/km)

Energy = abs(P ./ v);

Energy = Energy + 30*randn(N,1);

%% Create dataset

EVdata = table(speed,acceleration,roadGrade,temperature,Energy);

disp('Sample Dataset:')
disp(EVdata(1:5,:))

%% =====================================================
% STEP 2: DATA VISUALIZATION
% =====================================================

figure('Name','Dataset Visualization')

subplot(2,2,1)
scatter(speed,Energy,'filled')
xlabel('Speed (km/h)')
ylabel('Energy')

subplot(2,2,2)
scatter(acceleration,Energy,'filled')
xlabel('Acceleration')

subplot(2,2,3)
scatter(roadGrade,Energy,'filled')
xlabel('Road Grade')

subplot(2,2,4)
scatter(temperature,Energy,'filled')
xlabel('Temperature')

sgtitle('EV Dataset Relationships')

%% =====================================================
% STEP 3: FEATURE MATRIX
% =====================================================

X = [speed acceleration roadGrade temperature];
Y = Energy;

%% =====================================================
% STEP 4: TRAIN TEST SPLIT
% =====================================================

cv = cvpartition(size(X,1),'HoldOut',0.2);

Xtrain = X(training(cv),:);
Ytrain = Y(training(cv));

Xtest = X(test(cv),:);
Ytest = Y(test(cv));

%% =====================================================
% STEP 5: LINEAR REGRESSION
% =====================================================

disp('Training Linear Regression Model')

linearModel = fitlm(Xtrain,Ytrain);

disp(linearModel)

Ypred_lin = predict(linearModel,Xtest);

%% =====================================================
% STEP 6: RANDOM FOREST REGRESSION
% =====================================================

disp('Training Random Forest Model')

numTrees = 150;

RFmodel = TreeBagger(numTrees,Xtrain,Ytrain,...
    'Method','regression',...
    'OOBPrediction','on',...
    'OOBPredictorImportance','on');

Ypred_RF = predict(RFmodel,Xtest);

if iscell(Ypred_RF)
    Ypred_RF = cellfun(@str2double,Ypred_RF);
end

%% =====================================================
% STEP 7: MODEL EVALUATION
% =====================================================

disp('Model Evaluation')

MAE_lin = mean(abs(Ytest - Ypred_lin));
MSE_lin = mean((Ytest - Ypred_lin).^2);

MAE_RF = mean(abs(Ytest - Ypred_RF));
MSE_RF = mean((Ytest - Ypred_RF).^2);

SS_res = sum((Ytest - Ypred_RF).^2);
SS_tot = sum((Ytest - mean(Ytest)).^2);

R2_RF = 1 - SS_res/SS_tot;

fprintf('\nLinear Regression MAE: %.2f\n',MAE_lin)
fprintf('Random Forest MAE: %.2f\n',MAE_RF)

fprintf('\nRandom Forest R2: %.3f\n',R2_RF)

%% =====================================================
% STEP 8: PREDICTION VISUALIZATION
% =====================================================

figure('Name','Prediction Accuracy')

scatter(Ytest,Ypred_RF,'filled')

hold on

plot([min(Ytest) max(Ytest)],...
     [min(Ytest) max(Ytest)],...
     'r','LineWidth',2)

grid on

xlabel('Actual Energy')
ylabel('Predicted Energy')

title('Random Forest Prediction Accuracy')

%% =====================================================
% STEP 9: FEATURE IMPORTANCE
% =====================================================

figure('Name','Feature Importance')

bar(RFmodel.OOBPermutedPredictorDeltaError)

xlabel('Feature')

ylabel('Importance Score')

title('Feature Importance for EV Energy Prediction')

xticklabels({'Speed','Acceleration','Road Grade','Temperature'})

grid on

%% =====================================================
% STEP 10: MODEL COMPARISON
% =====================================================

figure('Name','Model Comparison')

bar([MAE_lin MAE_RF])

set(gca,'XTickLabel',{'Linear Regression','Random Forest'})

ylabel('MAE')

title('Model Comparison')

grid on

disp('===============================================')
disp('Demo Completed Successfully')
disp('===============================================')