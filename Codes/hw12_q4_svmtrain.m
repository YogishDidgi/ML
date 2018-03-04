%function hw12_q4_svmtrain
close all;
clear all;
load digitsData.mat;
regularizerCMin = 0.00001;
regularizerCMax = 50;%0.0001;
precision = 0.1;
regularizerC = [0.00001:0.00001:0.0001 0.0002:0.0001:0.001 0.002:0.001:0.01 0.02:0.01:0.1 0.2:0.1:1 2:1:10 20:10:50];%regularizerCMin:precision:regularizerCMax;
kernelFunction = @(arg1,arg2) (power((1 + arg1*arg2'),8));
%kernelFunction
%'polynomial','polyorder',8
figure,
SVMStruct = svmtrain(DataTrain_X,DataTrain_Y,'method','QP','boxconstraint',regularizerCMin,'kernel_function',kernelFunction,'ShowPlot',true);
xlabel('x1');
ylabel('x2');
title('decision boundary for Cmin');

figure,
SVMStruct = svmtrain(DataTrain_X,DataTrain_Y,'method','QP','boxconstraint',regularizerCMax,'kernel_function',kernelFunction,'ShowPlot',true);
xlabel('x1');
ylabel('x2');
title('decision boundary for Cmax');

%%
%5-fold cross-validation
cvFold = 10;
numSamplesEachSet = numTrainSamples/cvFold;

samplesSetX = zeros(numSamplesEachSet,size(DataTrain_X,2),cvFold);
samplesSetY = zeros(numSamplesEachSet,size(DataTrain_Y,2),cvFold);

sampleIndex = datasample(1:1:numTrainSamples,numTrainSamples,'Replace',false);
for i = 1:cvFold
    startIndex = (i - 1)*numSamplesEachSet + 1;
    endIndex = startIndex + numSamplesEachSet - 1;
    samplesSetX(:,:,i) = DataTrain_X(sampleIndex(startIndex:endIndex),:);
    samplesSetY(:,:,i) = DataTrain_Y(sampleIndex(startIndex:endIndex),:);
end

%%
%cross validation
Ecv = zeros(size(regularizerC,2),1);

for i = 1:size(regularizerC,2)
    fprintf('regulariserC: %f\t',regularizerC(i));
    tic
    for iterCV = 1:cvFold
        testX = samplesSetX(:,:,iterCV);
        testY = samplesSetY(:,:,iterCV);
        trainX = reshape(samplesSetX(:,:,1:end ~= iterCV),numTrainSamples - numSamplesEachSet,size(DataTrain_X,2));
        trainY = reshape(samplesSetY(:,:,1:end ~= iterCV),numTrainSamples - numSamplesEachSet,size(DataTrain_Y,2));
        
        SVMStruct = svmtrain(trainX,trainY,'method','QP','boxconstraint',regularizerC(i),'kernel_function',kernelFunction,'ShowPlot',false);
        testYSVM = svmclassify(SVMStruct,testX);
        Ecv(i) = Ecv(i) + sum(testYSVM ~= testY);
    end
    Ecv(i) = Ecv(i)/numTrainSamples;
    t1 = toc;
    fprintf('took %f sec\n',t1);
end
figure,
semilogx(regularizerC,Ecv);
xlabel('regularizer');
ylabel('Ecv');
title('regularizer vs Ecv');

%%
figure,
SVMStruct = svmtrain(DataTrain_X,DataTrain_Y,'method','QP','boxconstraint',regularizerCMin,'kernel_function',kernelFunction,'ShowPlot',true);
testYSVM = svmclassify(SVMStruct,DataTest_X);
Etest = sum(DataTest_Y ~= testYSVM)/numTestSamples

posLabel = find(testYSVM == 1);
negLabel = find(testYSVM == -1);
hold on,
% plot(DataTest_X(posLabel,1),DataTest_X(posLabel,2),'y+'),hold on
% plot(DataTest_X(negLabel,1),DataTest_X(negLabel,2),'go'),hold off
scatter(DataTest_Feature1_Not1, DataTest_Feature2_Not1, 'Marker', 'o', 'MarkerEdgeColor', 'g');hold on
scatter(DataTest_Feature1_1, DataTest_Feature2_1, 'Marker', '+', 'MarkerEdgeColor', 'y');hold off
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
%legend('y = +1','y = -1','TrainDataset - Not 1','TrainDataset - 1');
title('SVM with bestC');
