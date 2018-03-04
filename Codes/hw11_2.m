% function hw11_2
clear all
close all

fileIDTrain = fopen('ZipDigits.train','r');
fileIDTest = fopen('ZipDigits.test','r');
formatSpec = '%f';
sizeATrain = [257, Inf];
sizeATest = [257, Inf];
ATrain = fscanf(fileIDTrain,formatSpec,sizeATrain);
ATest = fscanf(fileIDTest,formatSpec,sizeATest);
fclose(fileIDTrain);
fclose(fileIDTest);
ATrain = ATrain';
ATest = ATest';
disp('Finished reading data');
%%
DataSet = [ATrain; ATest];
DataSet_y = -1*ones(size(DataSet,1),1);
Mask_1 = find(DataSet(:,1) == 1.0);
DataSet_y(Mask_1) = 1.0;
DataSet_Unlabelled = DataSet(:,2:257);
DataSet_Feature1 = sum(DataSet_Unlabelled, 2)/256;

DataSet_Unlabelled_Reshape = reshape(DataSet_Unlabelled,[],16,16);
[m n o] = size(DataSet_Unlabelled_Reshape);
DataSet_Feature2 = zeros(m,1);
for i=1:m
    sample_1(1:16,:) = DataSet_Unlabelled_Reshape(i,1:16,:);
    sample_1 = sample_1';
%     sample_1_rot = rot90(sample_1,2);
    diff_1 = abs(sample_1 - fliplr(sample_1));
    diff_2 = abs(sample_1 - flipud(sample_1));
%     diff_3 = abs(sample_1 - sample_1_rot);
    DataSet_Feature2(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

DataSet_Feature1_Normalised = mapminmax(DataSet_Feature1',-1,1)';
DataSet_Feature2_Normalised = mapminmax(DataSet_Feature2',-1,1)';
disp('Finished pre-processing data');
%%
numTrainSamples = 300;
numTestSamples = m - numTrainSamples;
[DataTrain_Feature1, DataTrainMask] = datasample(DataSet_Feature1_Normalised,numTrainSamples,1,'Replace',false);
DataTrainMask = DataTrainMask';
%DataTrainMask = randi(size(DataSet,1),numTrainSamples,1);
%DataTrain_Feature1 = DataSet_Feature1(DataTrainMask);
DataTrain_Feature2 = DataSet_Feature2_Normalised(DataTrainMask);
DataTrain_y = DataSet_y(DataTrainMask);

DummyMask = ones(size(DataSet,1),1);
DummyMask(DataTrainMask) = 0;
DataTest_Feature1 = DataSet_Feature1_Normalised(find(DummyMask));
DataTest_Feature2 = DataSet_Feature2_Normalised(find(DummyMask));
DataTest_y = DataSet_y(find(DummyMask));
disp('Done with data splitting');

%%
DataTrain_1_Mask = find(DataTrain_y == 1.0);
DataTrain_Not1_Mask = find(DataTrain_y == -1.0);
DataTrain_Feature1_1 = DataTrain_Feature1(DataTrain_1_Mask);
DataTrain_Feature1_Not1 = DataTrain_Feature1(DataTrain_Not1_Mask);
DataTrain_Feature2_1 = DataTrain_Feature2(DataTrain_1_Mask);
DataTrain_Feature2_Not1 = DataTrain_Feature2(DataTrain_Not1_Mask);

DataTest_1_Mask = find(DataTest_y == 1.0);
DataTest_Not1_Mask = find(DataTest_y == -1.0);
DataTest_Feature1_1 = DataTest_Feature1(DataTest_1_Mask);
DataTest_Feature1_Not1 = DataTest_Feature1(DataTest_Not1_Mask);
DataTest_Feature2_1 = DataTest_Feature2(DataTest_1_Mask);
DataTest_Feature2_Not1 = DataTest_Feature2(DataTest_Not1_Mask);

DataTrain_X = [DataTrain_Feature1 DataTrain_Feature2];
DataTest_X = [DataTest_Feature1 DataTest_Feature2];
DataTrain_Y = DataTrain_y;
DataTest_Y = DataTest_y;
%%
%leave one out cross validation for each k

errorLOOCV = zeros(numTrainSamples, 1);
for iter_k = 1:numTrainSamples
    r = 2/sqrt(iter_k);
    error = 0;
    fprintf('Evaluating for k = %d clusters;\t',iter_k);
    tic
    for iter_cv = 1:numTrainSamples
        %Leave one out
        data_x = DataTrain_X(1:end ~=iter_cv,:);
        data_y = DataTrain_Y(1:end ~=iter_cv,:);
        test_x = DataTrain_X(iter_cv,:);
        test_y = DataTrain_Y(iter_cv,:);
        %do clustering
        [clusterID, clusterMean] = kmeans(data_x,iter_k);
        %transform to z-space
        matrixZ = hw11_2_computeTransform(data_x, clusterMean, r);
        weight = linearRegression(matrixZ, data_y);
        test_z = hw11_2_computeTransform(test_x, clusterMean, r);
        error = error + evaluateError(test_z, weight, test_y);
    end
    t1 = toc;
    fprintf('Took %f seconds\n',t1);
    errorLOOCV(iter_k) = error/numTrainSamples;
end
%%
[minVal, minIndex] = min(errorLOOCV);
bestK = minIndex;
fprintf('bestK: %d\n',bestK);
kRange = 1:numTrainSamples-1;
figure(1),
plot(kRange,errorLOOCV);
xlabel('K');
ylabel('Ecv');
legend('error-LOOCV');
title('K vs Ecv');
%%

%%
%Calculate Ein, Etest
rBest = 2/sqrt(bestK);
Ein = 0;
Etest = 0;
Ecv = errorLOOCV(bestK);

data_x = DataTrain_X;
data_y = DataTrain_Y;
test_x = DataTrain_X;
test_y = DataTrain_Y;
%do clustering
[clusterID, clusterMeanBest] = kmeans(data_x,bestK);
%transform to z-space
matrixZ = hw11_2_computeTransform(data_x, clusterMeanBest, rBest);
weightBest = linearRegression(matrixZ, data_y);

test_z = hw11_2_computeTransform(test_x, clusterMeanBest, rBest);
Ein = evaluateError(test_z, weightBest, test_y);

test_x = DataTest_X;
test_y = DataTest_Y;

test_z = hw11_2_computeTransform(test_x, clusterMeanBest, rBest);
Etest = evaluateError(test_z, weightBest, test_y);

fprintf('Ein: %f\tEcv: %f\tEtest: %f\n', Ein, Ecv, Etest);

%%
%Plot decision boundary
xMin = -1;
xMax = 1;
yMin = -1;
yMax = 1;
precision = 0.01;%0.01;
[testx,testy] = meshgrid(xMin:precision:xMax,yMin:precision:yMax);
testx = reshape(testx,[],1);
testy = reshape(testy,[],1);
Data_test = [testx, testy];

test_z = hw11_2_computeTransform(Data_test, clusterMeanBest, rBest);
Label_test = sign(test_z*weightBest);

posLabel = find(Label_test == 1);
negLabel = find(Label_test == -1);

figure(2),
plot(testx(posLabel),testy(posLabel),'y+'),hold on
plot(testx(negLabel),testy(negLabel),'go'),hold on

scatter(DataTrain_Feature1_Not1, DataTrain_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTrain_Feature1_1, DataTrain_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold off

xlabel('Avg. Intensity');
ylabel('Avg. Difference');
legend('y = +1','y = -1','TrainDataset - Not 1','TrainDataset - 1');
title('bestK-RBF');

figure(3),
plot(testx(posLabel),testy(posLabel),'y+'),hold on
plot(testx(negLabel),testy(negLabel),'go'),hold on

scatter(DataTest_Feature1_Not1, DataTest_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTest_Feature1_1, DataTest_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold off

xlabel('Avg. Intensity');
ylabel('Avg. Difference');
legend('y = +1','y = -1','TestDataset - Not 1','TestDataset - 1');
title('bestK-RBF');
