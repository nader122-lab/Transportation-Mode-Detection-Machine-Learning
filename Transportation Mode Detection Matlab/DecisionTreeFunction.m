function [trainedClassifier, validationAccuracy] = trainClassifier(rawSensorDataTrain)
% [trainedClassifier, validationAccuracy] = trainClassifier(rawSensorDataTrain)
% Returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      rawSensorDataTrain: A table containing the same predictor and response
%       columns as those imported into the app.
%
%  Output:
%      trainedClassifier: A struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: A function to make predictions on new
%       data.
%
%      validationAccuracy: A double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument rawSensorDataTrain.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 16-Mar-2021 15:40:52


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = rawSensorDataTrain;
predictorNames = {'AXmean', 'AYmean', 'AZmean', 'AXstd', 'AYstd', 'AZstd', 'AXmedian', 'AYmedian', 'AZmedian', 'AXfreqEnergy', 'AYfreqEnergy', 'AZfreqEnergy', 'AXfreqMean', 'AYfreqMean', 'AZfreqMean', 'AvXmean', 'AvYmean', 'AvZmean', 'AvXstd', 'AvYstd', 'AvZstd', 'MXmean', 'MYmean', 'MZmean', 'MXstd', 'MYstd', 'MZstd'};
predictors = inputTable(:, predictorNames);
response = inputTable.Labelmean;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 100, ...
    'Surrogate', 'off', ...
    'ClassNames', [1; 2; 3; 4; 5]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'AXfreqEnergy', 'AXfreqMean', 'AXmean', 'AXmedian', 'AXstd', 'AYfreqEnergy', 'AYfreqMean', 'AYmean', 'AYmedian', 'AYstd', 'AZfreqEnergy', 'AZfreqMean', 'AZmean', 'AZmedian', 'AZstd', 'AvXmean', 'AvXstd', 'AvYmean', 'AvYstd', 'AvZmean', 'AvZstd', 'MXmean', 'MXstd', 'MYmean', 'MYstd', 'MZmean', 'MZstd'};
trainedClassifier.ClassificationTree = classificationTree;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = rawSensorDataTrain;
predictorNames = {'AXmean', 'AYmean', 'AZmean', 'AXstd', 'AYstd', 'AZstd', 'AXmedian', 'AYmedian', 'AZmedian', 'AXfreqEnergy', 'AYfreqEnergy', 'AZfreqEnergy', 'AXfreqMean', 'AYfreqMean', 'AZfreqMean', 'AvXmean', 'AvYmean', 'AvZmean', 'AvXstd', 'AvYstd', 'AvZstd', 'MXmean', 'MYmean', 'MZmean', 'MXstd', 'MYstd', 'MZstd'};
predictors = inputTable(:, predictorNames);
response = inputTable.Labelmean;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Set up holdout validation
cvp = cvpartition(response, 'Holdout', 0.2);
trainingPredictors = predictors(cvp.training, :);
trainingResponse = response(cvp.training, :);
trainingIsCategoricalPredictor = isCategoricalPredictor;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    trainingPredictors, ...
    trainingResponse, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 100, ...
    'Surrogate', 'off', ...
    'ClassNames', [1; 2; 3; 4; 5]);

% Create the result struct with predict function
treePredictFcn = @(x) predict(classificationTree, x);
validationPredictFcn = @(x) treePredictFcn(x);

% Add additional fields to the result struct


% Compute validation predictions
validationPredictors = predictors(cvp.test, :);
validationResponse = response(cvp.test, :);
[validationPredictions, validationScores] = validationPredictFcn(validationPredictors);

% Compute validation accuracy
correctPredictions = (validationPredictions == validationResponse);
isMissing = isnan(validationResponse);
correctPredictions = correctPredictions(~isMissing);
validationAccuracy = sum(correctPredictions)/length(correctPredictions);
