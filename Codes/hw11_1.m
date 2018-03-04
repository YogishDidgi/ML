% function hw11_1
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
kRange = 1:2:(numTrainSamples - 1);
errorLOOCV = zeros(length(kRange), 1);
for iter = 1:length(kRange)%299 KNN; because test samples are part of training set
    iter_knn = kRange(iter);
    Data_test = DataTrain_X;
    idxKnearest = knnsearch(DataTrain_X,Data_test,'K',iter_knn+1);%bcoz i 
    % am not removing the test sample from training set. So i have to get
    % the next nearest match
    idxKnearest_adjusted = idxKnearest(:,2:iter_knn+1);
    labelKnearest = DataTrain_Y(idxKnearest_adjusted);
    label_test = sign(sum(labelKnearest,2));
    errorLOOCV(iter) = (sum(DataTrain_Y ~= label_test))/numTrainSamples;
end
[minVal, minIndex] = min(errorLOOCV);
bestK = kRange(minIndex);
fprintf('bestK: %d\n',bestK);

figure(1),
plot(kRange,errorLOOCV);
xlabel('K');
ylabel('Ecv');
legend('error-LOOCV');
title('K vs Ecv');
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

idxKnearest = knnsearch(DataTrain_X,Data_test,'K',bestK);
idx_Label = zeros(size(idxKnearest));

Label_test = sign(sum(DataTrain_Y(idxKnearest),2));
posLabel = find(Label_test == 1);
negLabel = find(Label_test == -1);

%plot directly
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
title('bestK-NN');

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
title('bestK-NN');
%%
%Error calculation
Data_test = DataTrain_X;
idxKnearest = knnsearch(DataTrain_X,Data_test,'K',bestK);
labelKnearest = DataTrain_Y(idxKnearest);
label_test = sign(sum(labelKnearest,2));
Ein = (sum(DataTrain_Y ~= label_test))/numTrainSamples;
disp(Ein);
Ecv = errorLOOCV(bestK);
disp(Ecv);
Data_test = DataTest_X;
idxKnearest = knnsearch(DataTrain_X,Data_test,'K',bestK);
labelKnearest = DataTrain_Y(idxKnearest);
label_test = sign(sum(labelKnearest,2));
Etest = (sum(DataTest_Y ~= label_test))/numTrainSamples;

fprintf('Ein: %f\tEcv: %f\tEtest: %f\n', Ein, Ecv, Etest);









