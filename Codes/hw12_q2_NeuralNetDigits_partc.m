function hw12_q2_NeuralNetDigits
load DigitsData.mat;

dataTrainX = DataTrain_X;
dataTrainY = DataTrain_Y;
dataTrainX(:,1) = mapminmax_mine(DataTrain_X(:,1),-1,1);
dataTrainX(:,2) = mapminmax_mine(DataTrain_X(:,2),-1,1);

%Split into train and validation set
sizeTest = 50;
sizeTrain = numTrainSamples - sizeTest;%250
[DataTrain_Feature1_reduced, DataTrainMask_reduced] = datasample(DataTrain_Feature1,sizeTrain,1,'Replace',false);
DataTrainMask_reduced = DataTrainMask_reduced';
DataTrain_Feature2_reduced = DataTrain_Feature2(DataTrainMask_reduced);
DataTrain_y_reduced = DataTrain_y(DataTrainMask_reduced);

DummyMask_reduced = ones(size(DataTrain_Feature1,1),1);
DummyMask_reduced(DataTrainMask_reduced) = 0;
DataTest_Feature1_reduced = DataTrain_Feature1(find(DummyMask_reduced));
DataTest_Feature2_reduced = DataTrain_Feature2(find(DummyMask_reduced));
DataTest_y_reduced = DataTrain_y(find(DummyMask_reduced));
disp('Done with data splitting again');

dataTrainX = [DataTrain_Feature1_reduced DataTrain_Feature2_reduced];
dataTrainY = DataTrain_y_reduced;
dataTestX = [DataTest_Feature1_reduced DataTest_Feature2_reduced];
dataTestY = DataTest_y_reduced;
numTrainPoints = sizeTrain;
numTestPoints = sizeTest;

numLayers = 2;
L0_numNodes = 2;
L1_numNodes = 10;
L2_numNodes = 1;

L1_S = zeros(L1_numNodes,1);
L1_X = zeros(L1_numNodes,1);
L1_W = zeros(L0_numNodes + 1,L1_numNodes);
L1_Wprev = zeros(L0_numNodes + 1,L1_numNodes);
L1_delta = zeros(L1_numNodes,1);
L1_grad = zeros(L0_numNodes + 1,L1_numNodes);

L2_S = zeros(L2_numNodes,1);
L2_X = zeros(L2_numNodes,1);
L2_W = zeros(L1_numNodes + 1,L2_numNodes);
L2_Wprev = zeros(L1_numNodes + 1,L2_numNodes);
L2_delta = zeros(L2_numNodes,1);
L2_grad = zeros(L1_numNodes + 1,L2_numNodes);

%initialization
% L1_W = randi(5)*rand(size(L1_W));
% L2_W = randi(5)*rand(size(L2_W));
% save('L1_W_Init.mat','L1_W');
% save('L2_W_Init.mat','L2_W');
load 'L1_W_Init2.mat';
load 'L2_W_Init2.mat';
L1_Wbest = L1_W;
L2_Wbest = L2_W;
if(1)
learningRate = 0.01;
learningRateOrig = learningRate;
learningRateAlpha = 1.05;
learningRateBeta = 0.9;
stopCriteriaIters = 100000;%100000;%2*1000000;
variableGradDescentFlag = 1;
SGD = 0;
lambda = 0.01/numTrainPoints;
Ein = zeros(stopCriteriaIters,1);
Etest = zeros(stopCriteriaIters,1);
bestEin = realmax;
iter = 0;
while(1)
    tic
    iter = iter + 1;
    L1_grad = zeros(size(L1_grad));
    L2_grad = zeros(size(L2_grad));
    if(SGD)
        randIndex = randi(numTrainPoints);
        [L1_grad, L2_grad, error] = neuralNetworkCalculation(dataTrainX(randIndex,:), dataTrainY(randIndex), L1_W, L2_W);
    else
        error = 0;
        for i = 1:numTrainPoints
            [L1_gradTemp, L2_gradTemp, errorTemp] = neuralNetworkCalculation(dataTrainX(i,:), dataTrainY(i), L1_W, L2_W);
            L1_grad = L1_grad + L1_gradTemp;
            L2_grad = L2_grad + L2_gradTemp;
            error = error + errorTemp*errorTemp;
        end
        L1_grad = (L1_grad/numTrainPoints) + (2*lambda*L1_W/numTrainPoints);
        L2_grad = (L2_grad/numTrainPoints) + (2*lambda*L2_W/numTrainPoints);
        error = (error/(numTrainPoints)) + (lambda/numTrainPoints)*(sum(sum(L1_W)) + sum(L2_W));
    end

    %WEIGHT UPDATE
if(variableGradDescentFlag)
    %VARIABLE LEARNING RATE
    L1_Wnew = L1_W - learningRate*L1_grad;
    L2_Wnew = L2_W - learningRate*L2_grad;
    if(SGD)
        [testDataLabel, testDataOutput] = classifyData(L1_W,L2_W,dataTrainX);
        error = sum((testDataOutput - dataTrainY).*(testDataOutput - dataTrainY))/numTrainPoints;
    end
    [testDataLabelnew, testDataOutputnew] = classifyData(L1_Wnew,L2_Wnew,dataTrainX);
    errornew = sum((testDataOutputnew - dataTrainY).*(testDataOutputnew - dataTrainY))/numTrainPoints;
    errornew = errornew + (lambda/numTrainPoints)*(sum(sum(L1_Wnew)) + sum(L2_Wnew));
%     if(learningRate < 1e-3)
%         learningRate = learningRateOrig;
%     end
    if(errornew < error)
        L1_W = L1_Wnew;
        L2_W = L2_Wnew;
        learningRate = learningRateAlpha*learningRate;
        Ein(iter) = errornew;
    else
        learningRate = learningRateBeta*learningRate;
        Ein(iter) = error;
    end
    bestEin = Ein(iter);
    [testDataLabel, testDataOutput] = classifyData(L1_W,L2_W,dataTestX);
    Etest(iter) = sum((testDataOutput - dataTestY).*(testDataOutput - dataTestY))/numTestPoints;
else
    %CONSTANT LEARNING RATE
    [testDataLabel, testDataOutput] = classifyData(L1_W,L2_W,dataTrainX);
    Ein(iter) = sum((testDataOutput - dataTrainY).*(testDataOutput - dataTrainY))/numTrainPoints;
    if(Ein(iter) < bestEin)
        bestEin = Ein(iter);
        L1_Wbest = L1_W;
        L2_Wbest = L2_W;
    end
    L1_Wprev = L1_W;
    L1_W = L1_W - learningRate*L1_grad;
    L2_Wprev = L2_W;
    L2_W = L2_W - learningRate*L2_grad;
end
    t1 = toc;
    fprintf('Iter: %d Ein: %f took %f sec lr: %f\n', iter, bestEin, t1, learningRate);
    if((iter == stopCriteriaIters) || (learningRate < 1e-9) || ((iter > 1) && (Etest(iter) > Etest(iter - 1))))
        if(variableGradDescentFlag)
            L1_Wbest = L1_W;
            L2_Wbest = L2_W;
        else
            L1_W = L1_Wprev;
            L2_W = L2_Wprev;
            fprintf('Best Ein:%f\n', bestEin);
        end
        break;
    end
end
%FIG 1
x=1:stopCriteriaIters;
figure(1),
loglog(x,Ein);hold on;
loglog(x,Etest);hold off;
end
%FIG 2
testData = getTestData();
[testDataLabel,testDataOutput] = classifyData(L1_Wbest,L2_Wbest,testData);
posLabelTest = find(testDataLabel == 1);
negLabelTest = find(testDataLabel == -1);
posLabelTrain = find(dataTrainY == 1);
negLabelTrain = find(dataTrainY == -1);
figure(2),
plot(testData(posLabelTest,1),testData(posLabelTest,2),'y+'),hold on
plot(testData(negLabelTest,1),testData(negLabelTest,2),'go'),hold on
scatter(dataTrainX(negLabelTrain,1), dataTrainX(negLabelTrain,2), 'Marker', 'x', 'MarkerEdgeColor', 'r');hold on
scatter(dataTrainX(posLabelTrain,1), dataTrainX(posLabelTrain,2), 'Marker', 'o', 'MarkerEdgeColor', 'b');hold off
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
legend('y = +1','y = -1','TrainDataset - Not 1','TrainDataset - 1');
title('Neural Network with 10 hidden units');
end

function testData = getTestData()
xMin = -1;
xMax = 1;
yMin = -1;
yMax = 1;
precision = 0.01;%0.01;
[testx,testy] = meshgrid(xMin:precision:xMax,yMin:precision:yMax);
testx = reshape(testx,[],1);
testy = reshape(testy,[],1);
testData = [testx, testy];
end

function [testDataLabel, testDataOutput] = classifyData(L1_W,L2_W,testData)
testDataLabel = zeros(size(testData,1),1);
testDataOutput = zeros(size(testData,1),1);
for i = 1:size(testData,1)
    L1_S = L1_W'*([1 testData(i,:)]');
    L1_X = tanh(L1_S);
    L2_S = L2_W'*([1; L1_X]);
    L2_X = (L2_S);
    testDataOutput(i) = L2_X;
    testDataLabel(i) = sign(L2_X);
end
end

function [L1_grad, L2_grad, error] = neuralNetworkCalculation(dataX, dataY, L1_W, L2_W)
%FWD PROP
L1_S = L1_W'*([1 dataX]');
L1_X = tanh(L1_S);
L2_S = L2_W'*([1; L1_X]);
L2_X = (L2_S);
error = ((L2_X) - dataY);
%BWD PROP
L2_delta = 2*error*(1);
thetaDash = ones(size(L1_X)) - L1_X.*L1_X;
prevDelta = repmat(L2_delta', size(L1_S,1), 1);
L1_delta = thetaDash.*(L2_W(2:end,:).*prevDelta);
%GRADIENT
L1_grad = ([1 dataX]')*(L1_delta');
L2_grad = ([1; L1_X])*(L2_delta');
end