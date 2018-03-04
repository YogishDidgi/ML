function hw12_q2_NeuralNetDigits
load DigitsData.mat;

dataTrainX = DataTrain_X;
dataTrainY = DataTrain_Y;
dataTrainX(:,1) = mapminmax_mine(DataTrain_X(:,1),-1,1);
dataTrainX(:,2) = mapminmax_mine(DataTrain_X(:,2),-1,1);


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
L1_W = randi(5)*rand(size(L1_W));
L2_W = randi(5)*rand(size(L2_W));
L1_Wbest = L1_W;
L2_Wbest = L2_W;
if(1)
learningRate = 0.001;
learningRateAlpha = 1.05;
learningRateBeta = 0.8;
stopCriteriaIters = 100000;%100000;%2*1000000;
Ein = zeros(stopCriteriaIters,1);
bestEin = realmax;
L1_Wbest = 0;
L2_Wbest = 0;
iter = 0;
while(1)
    tic
    iter = iter + 1;
    L1_grad = zeros(size(L1_grad));
    L2_grad = zeros(size(L2_grad));
    randIndex = randi(numTrainSamples);
    %FWD PROP
    L1_S = L1_W'*([1 dataTrainX(randIndex,:)]');
    L1_X = tanh(L1_S);
    L2_S = L2_W'*([1; L1_X]);
    L2_X = (L2_S);
    error = ((L2_X) - dataTrainY(randIndex));
    Ein(iter) = Ein(iter) + error.*error;
    %BWD PROP
    L2_delta = 2*error*(1);
    thetaDash = ones(size(L1_X)) - L1_X.*L1_X;
    prevDelta = repmat(L2_delta', L1_numNodes, 1);
    L1_delta = thetaDash.*(L2_W(2:end,:).*prevDelta);
    %GRADIENT
    L1_grad = L1_grad + ([1 dataTrainX(randIndex,:)]')*(L1_delta');
    L2_grad = L2_grad + ([1; L1_X])*(L2_delta');
    if(Ein(iter) < bestEin)
        bestEin = Ein(iter);
        L1_Wbest = L1_W;
        L2_Wbest = L2_W;
    end
    %WEIGHT UPDATE
if(0)
    %VARIABLE LEARNING RATE
    
else
    %CONSTANT LEARNING RATE
    L1_Wprev = L1_W;
    L1_W = L1_W - learningRate*L1_grad;
    L2_Wprev = L2_W;
    L2_W = L2_W - learningRate*L2_grad;
end
    t1 = toc;
    fprintf('Iter: %d Ein: %f took %f sec\n', iter, Ein(iter), t1);
    if(iter == stopCriteriaIters)
        L1_W = L1_Wprev;
        L2_W = L2_Wprev;
        break;
    end
end
fprintf('Best Ein:%f\n', bestEin);
%FIG 1
x=1:stopCriteriaIters;
figure(1),
loglog(x,Ein);
end
%FIG 2
testData = getTestData();
testDataLabel = classifyData(L1_Wbest,L2_Wbest,testData);
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

function testDataLabel = classifyData(L1_W,L2_W,testData)
testDataLabel = zeros(size(testData,1),1);
for i = 1:size(testData,1)
    L1_S = L1_W'*([1 testData(i,:)]');
    L1_X = tanh(L1_S);
    L2_S = L2_W'*([1; L1_X]);
    L2_X = (L2_S);
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
prevDelta = repmat(L2_delta', L1_numNodes, 1);
L1_delta = thetaDash.*(L2_W(2:end,:).*prevDelta);
%GRADIENT
L1_grad = ([1 dataX]')*(L1_delta');
L2_grad = ([1; L1_X])*(L2_delta');
end