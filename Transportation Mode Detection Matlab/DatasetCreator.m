%Reshape data into timeframe of 5 seconds so 50 samples 21 total hours 
%Rows randomised aswell so we do not have a whole consecutive hour of each
%mode data
clear all
set(0,'DefaultFigureVisible','off')
A1 = dlmread('DataLabel.txt'); 
A2 = dlmread('AccelerationX.txt'); 
A3 = dlmread('AccelerationY.txt');
A4 = dlmread('AccelerationZ.txt'); 
A5 = dlmread('AngularVelocityX.txt');
A6 = dlmread('AngularVelocityY.txt'); 
A7 = dlmread('AngularVelocityZ.txt'); 
A8 = dlmread('MagneticFieldX.txt');
A9 = dlmread('MagneticFieldY.txt'); 
A10 = dlmread('MagneticFieldY.txt'); 

B1 = reshape(A1, [], 15120)'; %Label
B2 = reshape(A2, [], 15120)'; %Acceleration X
B3 = reshape(A3, [], 15120)'; %Acceleration Y
B4 = reshape(A4, [], 15120)'; %Acceleration Z
B5 = reshape(A5, [], 15120)'; %AngularVelocity X
B6 = reshape(A6, [], 15120)'; %AngularVelocity Y
B7 = reshape(A7, [], 15120)'; %acceleration Z
B8 = reshape(A8, [], 15120)'; %acceleration X
B9 = reshape(A9, [], 15120)'; %acceleration X
B10 = reshape(A10, [], 15120)'; %acceleration X

DataTable = table(B1, B2, B3, B4, B5, B6, B7, B8, B9, B10); % table full of sensor values
RandomTable1 = DataTable(randperm(size(DataTable, 1)),:); %Randomised Rows for table
RandomTable2 = tail(RandomTable1, 3024);
RandomTable3 = tail(RandomTable2, 2016);
TrainTable = head(RandomTable1, 12096); % train data sensor table 
TestTable1 = head(RandomTable2, 1008); %test table 1 sensor table
TestTable2 = head(RandomTable3, 1008); %test table 2 sensor table
TestTable3 = tail(RandomTable3, 1008); %test table 3 sensor table

Label = TrainTable.B1; % Randomised Label
AccX =  TrainTable.B2; %Randomised Acceleration X
AccY = TrainTable.B3; %Randomised Acceleration Y
AccZ =  TrainTable.B4; %Randomised Acceleration Z
AngVelX =  TrainTable.B5; %Randomised AngularVelocity X
AngVelY = TrainTable.B6; %Randomised AngularVelocity Y
AngVelZ = TrainTable.B7; %Randomised AngularVelocity Z
MagX = TrainTable.B7; %Randomised MagneticField X
MagY =  TrainTable.B8; %Random MagneticField Y
MagZ = TrainTable.B10; %Random Magneticfield Z

Label1 = TestTable1.B1; % Randomised Label
AccX1 =  TestTable1.B2; %Randomised Acceleration X
AccY1 = TestTable1.B3; %Randomised Acceleration Y
AccZ1 =  TestTable1.B4; %Randomised Acceleration Z
AngVelX1 =  TestTable1.B5; %Randomised AngularVelocity X
AngVelY1 = TestTable1.B6; %Randomised AngularVelocity Y
AngVelZ1 = TestTable1.B7; %Randomised AngularVelocity Z
MagX1 = TestTable1.B7; %Randomised MagneticField X
MagY1 =  TestTable1.B8; %Random MagneticField Y
MagZ1 = TestTable1.B10; %Random Magneticfield Z

Label2 = TestTable2.B1; % Randomised Label
AccX2 =  TestTable2.B2; %Randomised Acceleration X
AccY2 = TestTable2.B3; %Randomised Acceleration Y
AccZ2 =  TestTable2.B4; %Randomised Acceleration Z
AngVelX2 =  TestTable2.B5; %Randomised AngularVelocity X
AngVelY2 = TestTable2.B6; %Randomised AngularVelocity Y
AngVelZ2 = TestTable2.B7; %Randomised AngularVelocity Z
MagX2 = TestTable2.B7; %Randomised MagneticField X
MagY2 =  TestTable2.B8; %Random MagneticField Y
MagZ2 = TestTable2.B10; %Random Magneticfield Z

Label3 = TestTable3.B1; % Randomised Label
AccX3 =  TestTable3.B2; %Randomised Acceleration X
AccY3 = TestTable3.B3; %Randomised Acceleration Y
AccZ3 =  TestTable3.B4; %Randomised Acceleration Z
AngVelX3 =  TestTable3.B5; %Randomised AngularVelocity X
AngVelY3 = TestTable3.B6; %Randomised AngularVelocity Y
AngVelZ3 = TestTable3.B7; %Randomised AngularVelocity Z
MagX3 = TestTable3.B7; %Randomised MagneticField X
MagY3 =  TestTable3.B8; %Random MagneticField Y
MagZ3 = TestTable3.B10; %Random Magneticfield Z

%% reshape sensor data for feature extraction

Lab = reshape(Label',[],1); %acceleration Label
AX = reshape(AccX',[],1); %acceleration X
AY = reshape(AccY',[],1); %acceleration Y
AZ = reshape(AccZ',[],1); %acceleration Z
AvX = reshape(AngVelX',[],1); %angular velocity X
AvY = reshape(AngVelY',[],1); %angular velocity Y
AvZ = reshape(AngVelZ',[],1); %angular velocity Z
MX = reshape(MagX',[],1); %magnetic field X
MY = reshape(MagY',[],1); %magnetic field Y
MZ = reshape(MagZ',[],1); %magnetic field Z


%% Feature Extraction

n = 50; % represents 5 seconds 10hz sampling frequency so 50 samples each sliding window frame size
Aaverage = sqrt(AX.^2+AY.^2+AZ.^2); 
Labelmean = arrayfun(@(i) mean(Lab(i:i+n-1)),1:n:length(Lab)-n+1)'; % the averaged vector
Amean = arrayfun(@(i) mean(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % the averaged vector
Astd = arrayfun(@(i) std(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % std vector
Amedian = arrayfun(@(i) median(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % median vector

y= 50;
x = 1;

for i = 1:12096
    G1 = fft(Aaverage(x:y)); % fourier transform for frequency domain analysis
    mean1 = abs(G1);
    energy1 = G1.*conj(G1);      
    x=y+1;
    y = y+50;
    AfreqEnergy(i,1) = mean(energy1); %frequency domain energy
    AfreqMean(i,1) = mean(mean1);  %frequency domain mean 
    AfreqMax(i,1) = max(mean1);    %max frequecy value
end   

AvXmean = arrayfun(@(i) mean(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % the averaged vector
AvYmean = arrayfun(@(i) mean(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % the averaged vector
AvZmean = arrayfun(@(i) mean(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % the averaged vector
AvXstd = arrayfun(@(i) std(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % std vector
AvYstd  = arrayfun(@(i) std(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % std vector
AvZstd = arrayfun(@(i) std(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % std vector
MXstd = arrayfun(@(i) std(MX(i:i+n-1)),1:n:length(MX)-n+1)'; % std vector
MYstd  = arrayfun(@(i) std(MY(i:i+n-1)),1:n:length(MY)-n+1)'; % std vector
MZstd = arrayfun(@(i) std(MZ(i:i+n-1)),1:n:length(MZ)-n+1)'; % std vector

%% Sensor Train Table 

rawSensorDataTrain = table(Labelmean, Amean, Astd, Amedian, AfreqEnergy, AfreqMean, AfreqMax, AvXmean, AvYmean, AvZmean, AvXstd, AvYstd, AvZstd, MXstd, MYstd, MZstd); % sensor data train for selected features

%% Train data Classification

%  Input:
%      rawSensorDataTrain: A table containing the same predictor and response
%       columns as those imported into the app.
%
%  Output:
%      trainedClassifier: A struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      validationAccuracy: A double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = rawSensorDataTrain;
predictorNames = {'Amean', 'Astd', 'Amedian', 'AfreqEnergy', 'AfreqMean', 'AfreqMax', 'AvXmean', 'AvYmean', 'AvZmean', 'AvXstd', 'AvYstd', 'AvZstd', 'MXstd', 'MYstd', 'MZstd'};
predictors = inputTable(:, predictorNames);
response = inputTable.Labelmean;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options with max number of splits set to 19
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [1; 2; 3; 4; 5]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));
trainedClassifier.RequiredVariables = {'AfreqEnergy', 'AfreqMax', 'AfreqMean', 'Amean', 'Amedian', 'Astd', 'AvXmean', 'AvXstd', 'AvYmean', 'AvYstd', 'AvZmean', 'AvZstd', 'MXstd', 'MYstd', 'MZstd'};
trainedClassifier.ClassificationTree = classificationTree;

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = rawSensorDataTrain;
predictorNames = {'Amean', 'Astd', 'Amedian', 'AfreqEnergy', 'AfreqMean', 'AfreqMax', 'AvXmean', 'AvYmean', 'AvZmean', 'AvXstd', 'AvYstd', 'AvZstd', 'MXstd', 'MYstd', 'MZstd'};
predictors = inputTable(:, predictorNames);
response = inputTable.Labelmean;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
%% Test Data 1 reshape sensor data

Lab = reshape(Label1',[],1); %acceleration Label
AX = reshape(AccX1',[],1); %acceleration X
AY = reshape(AccY1',[],1); %acceleration Y
AZ = reshape(AccZ1',[],1); %acceleration Z
AvX = reshape(AngVelX1',[],1); %angular velocity X
AvY = reshape(AngVelY1',[],1); %angular velocity Y
AvZ = reshape(AngVelZ1',[],1); %angular velocity Z
MX = reshape(MagX1',[],1); %magnetic field X
MY = reshape(MagY1',[],1); %magnetic field Y
MZ = reshape(MagZ1',[],1); %magnetic field Z

%% Test data 1 Feature Extraction

n = 50; % represents 5 seconds 10hz sampling frequency so 50 samples each sliding window frame size
Aaverage = sqrt(AX.^2+AY.^2+AZ.^2);
Labelmean = arrayfun(@(i) mean(Lab(i:i+n-1)),1:n:length(Lab)-n+1)'; % the averaged vector
Amean = arrayfun(@(i) mean(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % the averaged vector
Astd = arrayfun(@(i) std(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % std vector
Amedian = arrayfun(@(i) median(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; %median vector


y= 50;
x = 1;

for i = 1:1008   
    G1 = fft(Aaverage(x:y)); % fourier transform for frequency domain analysis
    mean1 = abs(G1);
    energy1 = G1.*conj(G1);      
    x=y+1;
    y = y+50;
    AfreqEnergy1(i,1) = mean(energy1); % frequency domain energy
    AfreqMean1(i,1) = mean(mean1);  % frequency domain mean
    AfreqMax1(i,1) = max(mean1);    % max frequency value
end   

AvXmean = arrayfun(@(i) mean(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % the averaged vector
AvYmean = arrayfun(@(i) mean(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % the averaged vector
AvZmean = arrayfun(@(i) mean(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % the averaged vector
AvXstd = arrayfun(@(i) std(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % std vector
AvYstd  = arrayfun(@(i) std(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % std vector
AvZstd = arrayfun(@(i) std(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % std vector
MXstd = arrayfun(@(i) std(MX(i:i+n-1)),1:n:length(MX)-n+1)'; % std vector
MYstd  = arrayfun(@(i) std(MY(i:i+n-1)),1:n:length(MY)-n+1)'; % std vector
MZstd = arrayfun(@(i) std(MZ(i:i+n-1)),1:n:length(MZ)-n+1)'; % std vector

%% Test Data 1 sensor test data table

rawSensorDataTest1 = table(Labelmean, Amean, Astd, Amedian, AfreqEnergy1, AfreqMean1, AfreqMax1, AvXmean, AvYmean, AvZmean, AvXstd, AvYstd, AvZstd, MXstd, MYstd, MZstd); % sensor data train for selected features
rawSensorDataTest1.Properties.VariableNames{5} = 'AfreqEnergy'; % rename variables for prediction comparison
rawSensorDataTest1.Properties.VariableNames{6} = 'AfreqMean';
rawSensorDataTest1.Properties.VariableNames{7} = 'AfreqMax';


%% Test Data 2 reshape

Lab = reshape(Label2',[],1); %acceleration Label
AX = reshape(AccX2',[],1); %acceleration X
AY = reshape(AccY2',[],1); %acceleration Y
AZ = reshape(AccZ2',[],1); %acceleration Z
AvX = reshape(AngVelX2',[],1); %angular velocity X
AvY = reshape(AngVelY2',[],1); %angular velocity Y
AvZ = reshape(AngVelZ2',[],1); %angular velocity Z
MX = reshape(MagX2',[],1); %magnetic field X
MY = reshape(MagY2',[],1); %magnetic field Y
MZ = reshape(MagZ2',[],1); %magnetic field Z

%% Test data 2 feature extraction

n = 50; % represents 5 seconds 10hz sampling frequency so 50 samples each sliding window frame size
Aaverage = sqrt(AX.^2+AY.^2+AZ.^2);
Labelmean = arrayfun(@(i) mean(Lab(i:i+n-1)),1:n:length(Lab)-n+1)'; % the averaged vector
Amean = arrayfun(@(i) mean(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % the averaged vector
Astd = arrayfun(@(i) std(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % std vector
Amedian = arrayfun(@(i) median(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % median vector


y= 50;
x = 1;

for i = 1:1008
    G1 = fft(Aaverage(x:y)); % fourier transform for frequency domain analysis
    mean1 = abs(G1);
    energy1 = G1.*conj(G1);      
    x=y+1;
    y = y+50;
    AfreqEnergy2(i,1) = mean(energy1);
    AfreqMean2(i,1) = mean(mean1); 
    AfreqMax2(i,1) = max(mean1);
end   

AvXmean = arrayfun(@(i) mean(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % the averaged vector
AvYmean = arrayfun(@(i) mean(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % the averaged vector
AvZmean = arrayfun(@(i) mean(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % the averaged vector
AvXstd = arrayfun(@(i) std(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % std vector
AvYstd  = arrayfun(@(i) std(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % std vector
AvZstd = arrayfun(@(i) std(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % std vector
MXstd = arrayfun(@(i) std(MX(i:i+n-1)),1:n:length(MX)-n+1)'; % std vector
MYstd  = arrayfun(@(i) std(MY(i:i+n-1)),1:n:length(MY)-n+1)'; % std vector
MZstd = arrayfun(@(i) std(MZ(i:i+n-1)),1:n:length(MZ)-n+1)'; % std vector

%% Test data 2

rawSensorDataTest2 = table(Labelmean, Amean, Astd, Amedian, AfreqEnergy2, AfreqMean2, AfreqMax2, AvXmean, AvYmean, AvZmean, AvXstd, AvYstd, AvZstd, MXstd, MYstd, MZstd); % sensor data train for selected features
rawSensorDataTest2.Properties.VariableNames{5} = 'AfreqEnergy'; % rename variables for prediction comparison 
rawSensorDataTest2.Properties.VariableNames{6} = 'AfreqMean';
rawSensorDataTest2.Properties.VariableNames{7} = 'AfreqMax';




%% Test data 3 reshape

Lab = reshape(Label3',[],1); %acceleration Label
AX = reshape(AccX3',[],1); %acceleration X
AY = reshape(AccY3',[],1); %acceleration Y
AZ = reshape(AccZ3',[],1); %acceleration Z
AvX = reshape(AngVelX3',[],1); %angular velocity X
AvY = reshape(AngVelY3',[],1); %angular velocity Y
AvZ = reshape(AngVelZ3',[],1); %angular velocity Z
MX = reshape(MagX3',[],1); %magnetic field X
MY = reshape(MagY3',[],1); %magnetic field Y
MZ = reshape(MagZ3',[],1); %magnetic field Z

%% Test Data 3 Feature Extraction

n = 50; % represents 5 seconds 10hz sampling frequency so 50 samples each sliding window frame size
Aaverage = sqrt(AX.^2+AY.^2+AZ.^2);
Labelmean = arrayfun(@(i) mean(Lab(i:i+n-1)),1:n:length(Lab)-n+1)'; % the averaged vector
Amean = arrayfun(@(i) mean(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % the averaged vector
Astd = arrayfun(@(i) std(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % std vector
Amedian = arrayfun(@(i) median(Aaverage(i:i+n-1)),1:n:length(Aaverage)-n+1)'; % median vector


y= 50;
x = 1;

for i = 1:1008
    
    G1 = fft(Aaverage(x:y)); % fourier transform for frequency domain analysis
    mean1 = abs(G1);
    energy1 = G1.*conj(G1);      
    x=y+1;
    y = y+50;
    AfreqEnergy3(i,1) = mean(energy1); % frequency domain energy
    AfreqMean3(i,1) = mean(mean1);  % frequency domain mean
    AfreqMax3(i,1) = max(mean1);    % max frequency value
end   

AvXmean = arrayfun(@(i) mean(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % the averaged vector
AvYmean = arrayfun(@(i) mean(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % the averaged vector
AvZmean = arrayfun(@(i) mean(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % the averaged vector
AvXstd = arrayfun(@(i) std(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % std vector
AvYstd  = arrayfun(@(i) std(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % std vector
AvZstd = arrayfun(@(i) std(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % std vector
MXstd = arrayfun(@(i) std(MX(i:i+n-1)),1:n:length(MX)-n+1)'; % std vector
MYstd  = arrayfun(@(i) std(MY(i:i+n-1)),1:n:length(MY)-n+1)'; % std vector
MZstd = arrayfun(@(i) std(MZ(i:i+n-1)),1:n:length(MZ)-n+1)'; % std vector

%% Test Data 3

rawSensorDataTest3 = table(Labelmean, Amean, Astd, Amedian, AfreqEnergy3, AfreqMean3, AfreqMax3, AvXmean, AvYmean, AvZmean, AvXstd, AvYstd, AvZstd, MXstd, MYstd, MZstd); % sensor data train for selected features
rawSensorDataTest3.Properties.VariableNames{5} = 'AfreqEnergy'; % rename variables for prediction comparison
rawSensorDataTest3.Properties.VariableNames{6} = 'AfreqMean';
rawSensorDataTest3.Properties.VariableNames{7} = 'AfreqMax';



%% Test Data 1 Classification

Predictions = trainedClassifier.predictFcn(rawSensorDataTest1); % test trained model on test dataset 1
AccuracyTest1 = sum ( rawSensorDataTest1.Labelmean==Predictions) / numel (rawSensorDataTest1.Labelmean)*100; % accuracy value for test data 
ConfusionMatrix = confusionmat(rawSensorDataTest1.Labelmean, Predictions); % confusion matrix for predictions 
ConfusionMatrixPng = confusionchart(rawSensorDataTest1.Labelmean, Predictions); 
saveas(ConfusionMatrixPng,'ConfusionMat','fig');
figure_handle = openfig('ConfusionMat.fig');
print(figure_handle,'-dpng','ConfusionMat1.png'); %save confusion matrix for use in web application
F11 = ConfusionMatrix(1,1)/(ConfusionMatrix(1,1)+ (0.5*((sum(ConfusionMatrix(1, :))-ConfusionMatrix(1,1)) + (sum(ConfusionMatrix(:, 1))-ConfusionMatrix(1,1))))); 
F12 = ConfusionMatrix(2,2)/(ConfusionMatrix(2,2)+ (0.5*((sum(ConfusionMatrix(2, :))-ConfusionMatrix(2,2)) + (sum(ConfusionMatrix(:, 2))-ConfusionMatrix(2,2)))));
F13 = ConfusionMatrix(3,3)/(ConfusionMatrix(3,3)+ (0.5*((sum(ConfusionMatrix(3, :))-ConfusionMatrix(3,3)) + (sum(ConfusionMatrix(:, 3))-ConfusionMatrix(3,3)))));
F14 = ConfusionMatrix(4,4)/(ConfusionMatrix(4,4)+ (0.5*((sum(ConfusionMatrix(4, :))-ConfusionMatrix(4,4)) + (sum(ConfusionMatrix(:, 4))-ConfusionMatrix(4,4)))));
F15 = ConfusionMatrix(5,5)/(ConfusionMatrix(5,5)+ (0.5*((sum(ConfusionMatrix(5, :))-ConfusionMatrix(5,5)) + (sum(ConfusionMatrix(:, 5))-ConfusionMatrix(5,5)))));
F1score1 = (F11 + F12 + F13 + F14 + F15)/5;

%% Test Data 2 Classification

Predictions2 = trainedClassifier.predictFcn(rawSensorDataTest2); % test trained model on test dataset 2
AccuracyTest2 = sum ( rawSensorDataTest2.Labelmean==Predictions2) / numel (rawSensorDataTest2.Labelmean)*100; % accuracy value for test data
ConfusionMatrix = confusionmat(rawSensorDataTest2.Labelmean, Predictions2); % confusion matrix for predictions
ConfusionMatrixPng = confusionchart(rawSensorDataTest2.Labelmean, Predictions2);
saveas(ConfusionMatrixPng,'ConfusionMat','fig');
figure_handle = openfig('ConfusionMat.fig');
print(figure_handle,'-dpng','ConfusionMat2.png'); %save confusion matrix for use in web application
F11 = ConfusionMatrix(1,1)/(ConfusionMatrix(1,1)+ (0.5*((sum(ConfusionMatrix(1, :))-ConfusionMatrix(1,1)) + (sum(ConfusionMatrix(:, 1))-ConfusionMatrix(1,1))))); 
F12 = ConfusionMatrix(2,2)/(ConfusionMatrix(2,2)+ (0.5*((sum(ConfusionMatrix(2, :))-ConfusionMatrix(2,2)) + (sum(ConfusionMatrix(:, 2))-ConfusionMatrix(2,2)))));
F13 = ConfusionMatrix(3,3)/(ConfusionMatrix(3,3)+ (0.5*((sum(ConfusionMatrix(3, :))-ConfusionMatrix(3,3)) + (sum(ConfusionMatrix(:, 3))-ConfusionMatrix(3,3)))));
F14 = ConfusionMatrix(4,4)/(ConfusionMatrix(4,4)+ (0.5*((sum(ConfusionMatrix(4, :))-ConfusionMatrix(4,4)) + (sum(ConfusionMatrix(:, 4))-ConfusionMatrix(4,4)))));
F15 = ConfusionMatrix(5,5)/(ConfusionMatrix(5,5)+ (0.5*((sum(ConfusionMatrix(5, :))-ConfusionMatrix(5,5)) + (sum(ConfusionMatrix(:, 5))-ConfusionMatrix(5,5)))));
F1score2 = (F11 + F12 + F13 + F14 + F15)/5; %F1 score value for test data



%% Test Data 3 Classification

Predictions = trainedClassifier.predictFcn(rawSensorDataTest3); % test trained model on test dataset 3
AccuracyTest3 = sum ( rawSensorDataTest3.Labelmean==Predictions) / numel (rawSensorDataTest3.Labelmean)*100; % accuracy value for test data
ConfusionMatrix = confusionmat(rawSensorDataTest3.Labelmean, Predictions); % confusion matrix for predictions
ConfusionMatrixPng = confusionchart(rawSensorDataTest3.Labelmean, Predictions);
saveas(ConfusionMatrixPng,'ConfusionMat','fig');
figure_handle = openfig('ConfusionMat.fig');
print(figure_handle,'-dpng','ConfusionMat3.png'); %save confusion matrix for use in web application
F11 = ConfusionMatrix(1,1)/(ConfusionMatrix(1,1)+ (0.5*((sum(ConfusionMatrix(1, :))-ConfusionMatrix(1,1)) + (sum(ConfusionMatrix(:, 1))-ConfusionMatrix(1,1))))); 
F12 = ConfusionMatrix(2,2)/(ConfusionMatrix(2,2)+ (0.5*((sum(ConfusionMatrix(2, :))-ConfusionMatrix(2,2)) + (sum(ConfusionMatrix(:, 2))-ConfusionMatrix(2,2)))));
F13 = ConfusionMatrix(3,3)/(ConfusionMatrix(3,3)+ (0.5*((sum(ConfusionMatrix(3, :))-ConfusionMatrix(3,3)) + (sum(ConfusionMatrix(:, 3))-ConfusionMatrix(3,3)))));
F14 = ConfusionMatrix(4,4)/(ConfusionMatrix(4,4)+ (0.5*((sum(ConfusionMatrix(4, :))-ConfusionMatrix(4,4)) + (sum(ConfusionMatrix(:, 4))-ConfusionMatrix(4,4)))));
F15 = ConfusionMatrix(5,5)/(ConfusionMatrix(5,5)+ (0.5*((sum(ConfusionMatrix(5, :))-ConfusionMatrix(5,5)) + (sum(ConfusionMatrix(:, 5))-ConfusionMatrix(5,5)))));
F1score3 = (F11 + F12 + F13 + F14 + F15)/5; %F1 score value for test data

%% Save mat file for workspace value for use in Web Application 

save('n.mat');

%% Launch Web App

ProjectApp



