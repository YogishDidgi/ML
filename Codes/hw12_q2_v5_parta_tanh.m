%function hw12_q2
load digitsData.mat;
dataTrainX = DataTrain_X;
dataTrainY = DataTrain_Y;
dataTrainX(:,1) = mapminmax_mine(DataTrain_X(:,1),-1,1);
dataTrainX(:,2) = mapminmax_mine(DataTrain_X(:,2),-1,1);
%%
NeuralNetwork = struct('numInputNodes',{},'numOutputNodes',{},'numLayers',{},'layerData',{});
NeuralNetworkLayer = struct('numHiddenNodes',{},'S',{},'X',{},'W',{},'delta',{},'gradient',{},'Wprev',{});
transformFunction = 'tanh';%'identity' or 'tanh'

%%
%Initialization
NeuralNetwork(1).numInputNodes = 2;%not including the constant 1 node
NeuralNetwork(1).numOutputNodes = 1;
NeuralNetwork(1).numLayers = 2;%including the output layer
NeuralNetwork(1).layerData = NeuralNetworkLayer;

%Layer 1 data
NeuralNetwork(1).layerData(1).numHiddenNodes = 10;
NeuralNetwork(1).layerData(1).S = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes,1);
NeuralNetwork(1).layerData(1).X = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes,1);%don't include the constant 1 node
NeuralNetwork(1).layerData(1).W = zeros(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);
NeuralNetwork(1).layerData(1).delta = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes,1);%same size as S
NeuralNetwork(1).layerData(1).gradient = zeros(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);
NeuralNetwork(1).layerData(1).Wprev = zeros(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);

%Layer 2 data
%This is also the output layer
NeuralNetwork(1).layerData(2).numHiddenNodes = 1;
NeuralNetwork(1).layerData(2).S = zeros(NeuralNetwork(1).layerData(2).numHiddenNodes,1);
NeuralNetwork(1).layerData(2).X = zeros(NeuralNetwork(1).layerData(2).numHiddenNodes,1);%don't include the constant 1 node
NeuralNetwork(1).layerData(2).W = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);
NeuralNetwork(1).layerData(2).delta = zeros(NeuralNetwork(1).layerData(2).numHiddenNodes,1);
NeuralNetwork(1).layerData(2).gradient = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);
NeuralNetwork(1).layerData(2).Wprev = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);

%%
%initialize weights randomly between randMin and randMax
% randMin = 0;
% randMax = 2;
% NeuralNetwork(1).layerData(1).W = randMin + (randMax - randMin)*rand(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);
% NeuralNetwork(1).layerData(2).W = randMin + (randMax - randMin)*rand(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);
NeuralNetwork(1).layerData(1).W = rand(size(NeuralNetwork(1).layerData(1).W));
NeuralNetwork(1).layerData(2).W = rand(size(NeuralNetwork(1).layerData(2).W));
%%
%Substitute parameters for the NN structure
NeuralNetwork_numInputNodes = NeuralNetwork(1).numInputNodes;
NeuralNetwork_numOutputNodes = NeuralNetwork(1).numOutputNodes;
NeuralNetwork_numLayers = NeuralNetwork(1).numLayers;

NeuralNetwork_layerData_1_numHiddenNodes = NeuralNetwork(1).layerData(1).numHiddenNodes;
NeuralNetwork_layerData_1_S = NeuralNetwork(1).layerData(1).S;
NeuralNetwork_layerData_1_X = NeuralNetwork(1).layerData(1).X;
NeuralNetwork_layerData_1_W = NeuralNetwork(1).layerData(1).W;
NeuralNetwork_layerData_1_delta = NeuralNetwork(1).layerData(1).delta;
NeuralNetwork_layerData_1_gradient = NeuralNetwork(1).layerData(1).gradient;
NeuralNetwork_layerData_1_Wprev = NeuralNetwork(1).layerData(1).Wprev;

NeuralNetwork_layerData_2_numHiddenNodes = NeuralNetwork(1).layerData(2).numHiddenNodes;
NeuralNetwork_layerData_2_S = NeuralNetwork(1).layerData(2).S;
NeuralNetwork_layerData_2_X = NeuralNetwork(1).layerData(2).X;
NeuralNetwork_layerData_2_W = NeuralNetwork(1).layerData(2).W;
NeuralNetwork_layerData_2_delta = NeuralNetwork(1).layerData(2).delta;
NeuralNetwork_layerData_2_gradient = NeuralNetwork(1).layerData(2).gradient;
NeuralNetwork_layerData_2_Wprev = NeuralNetwork(1).layerData(2).Wprev;

%%
%Variable learning rate gradient descent
learningRate = 0.01;
learningRatePrev = 0.01;
learningRateAlpha = 1.05;
learningRateBeta = 0.8;
stopCriteriaIters = 1000;%2*1000000;
Ein = zeros(stopCriteriaIters,1);
Ein_prev = realmax;
iters = 0;
numTrainPoints = numTrainSamples;
while(1)
    tic
    iters = iters + 1;
    %initialization
%     Ein(iters) = 0;
    NeuralNetwork_layerData_1_gradient = zeros(size(NeuralNetwork_layerData_1_W));
    NeuralNetwork_layerData_2_gradient = zeros(size(NeuralNetwork_layerData_2_W));
    %Gradient computation
    for i = 1:numTrainPoints
        %Forward prop
        NeuralNetwork_layerData_1_S = ((NeuralNetwork_layerData_1_W)')*([1;dataTrainX(i,:)']);
        NeuralNetwork_layerData_1_X = tanh(NeuralNetwork_layerData_1_S);
        NeuralNetwork_layerData_2_S = ((NeuralNetwork_layerData_2_W)')*([1;NeuralNetwork_layerData_1_X]);
        NeuralNetwork_layerData_2_X = (NeuralNetwork_layerData_2_S);
        error = (NeuralNetwork_layerData_2_X - dataTrainY(i));
        
        Ein(iters) = Ein(iters) + error.*error;
        %Backward prop
        NeuralNetwork_layerData_2_delta = 2*(error)*(1);
        thetaDash = ones(size(NeuralNetwork_layerData_1_X)) - NeuralNetwork_layerData_1_X.*NeuralNetwork_layerData_1_X;
        prevDelta = repmat(NeuralNetwork_layerData_2_delta', NeuralNetwork_layerData_1_numHiddenNodes, 1);
        NeuralNetwork_layerData_1_delta = thetaDash.*(NeuralNetwork_layerData_2_W(2:end,:).*prevDelta);
        
        %Gradient calc
        NeuralNetwork_layerData_1_gradient = NeuralNetwork_layerData_1_gradient + ([1; dataTrainX(i,:)'])*(NeuralNetwork_layerData_1_delta');
        NeuralNetwork_layerData_2_gradient = NeuralNetwork_layerData_2_gradient + ([1; NeuralNetwork_layerData_1_X])*(NeuralNetwork_layerData_2_delta');
    end
    %Normalization of error and accumulated gradients; Also Weights updation
    Ein(iters) = Ein(iters)/(4*numTrainPoints);
    NeuralNetwork_layerData_1_gradient = NeuralNetwork_layerData_1_gradient/numTrainPoints;
    NeuralNetwork_layerData_2_gradient = NeuralNetwork_layerData_2_gradient/numTrainPoints;
   
    if(iters > 1)
        if(Ein(iters) < Ein(iters - 1))
            %Accept the update
            learningRatePrev = learningRate;
            learningRate = learningRateAlpha*learningRate;
            NeuralNetwork_layerData_1_Wprev = NeuralNetwork_layerData_1_W;
            NeuralNetwork_layerData_2_Wprev = NeuralNetwork_layerData_2_W;
            NeuralNetwork_layerData_1_W = NeuralNetwork_layerData_1_W - learningRate*NeuralNetwork_layerData_1_gradient;
            NeuralNetwork_layerData_2_W = NeuralNetwork_layerData_2_W - learningRate*NeuralNetwork_layerData_2_gradient;
        else
            %Reject the update and set all variables to previous state
            Ein(iters) = Ein(iters - 1);
            learningRate = learningRateBeta*learningRatePrev;
            learningRatePrev = learningRate;
            NeuralNetwork_layerData_1_W = NeuralNetwork_layerData_1_Wprev - learningRate*NeuralNetwork_layerData_1_gradient;
            NeuralNetwork_layerData_2_W = NeuralNetwork_layerData_2_Wprev - learningRate*NeuralNetwork_layerData_2_gradient;
            NeuralNetwork_layerData_1_Wprev = NeuralNetwork_layerData_1_W;
            NeuralNetwork_layerData_2_Wprev = NeuralNetwork_layerData_2_W;
        end
    else
        learningRatePrev = learningRate;
        %Update weights to original learning rate
        NeuralNetwork_layerData_1_Wprev = NeuralNetwork_layerData_1_W;
        NeuralNetwork_layerData_2_Wprev = NeuralNetwork_layerData_2_W;
        NeuralNetwork_layerData_1_W = NeuralNetwork_layerData_1_W - learningRate*NeuralNetwork_layerData_1_gradient;
        NeuralNetwork_layerData_2_W = NeuralNetwork_layerData_2_W - learningRate*NeuralNetwork_layerData_2_gradient;
    end
    
    t1 = toc;
    fprintf('Iter: %d Ein: %f took %f sec lr: %f\n', iters, Ein(iters), t1, learningRate);
    if(iters == stopCriteriaIters || learningRate < 1e-6)
        NeuralNetwork_layerData_1_W = NeuralNetwork_layerData_1_Wprev;
        NeuralNetwork_layerData_2_W = NeuralNetwork_layerData_2_Wprev;
        break;
    end
end
%Use prev Weights after loop break *************** IMP  IMP  IMP  IMP  IMP  IMP **********************
fprintf('After gradient descent: Ein: %f numIters: %d\n', Ein(iters), iters);

%%
x=1:stopCriteriaIters;
figure(1),
%semilogy(x,Ein);
loglog(x,Ein);

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
Data_test_label = zeros(size(testx,1),1);
% NeuralNetwork_layerData_1_W = NeuralNetwork(1).layerData(1).W;
% NeuralNetwork_layerData_2_W = NeuralNetwork(1).layerData(2).W;

for i = 1:size(testx,1)
    NeuralNetwork_layerData_1_S = ((NeuralNetwork_layerData_1_W)')*([1;Data_test(i,:)']);
    NeuralNetwork_layerData_1_X = tanh(NeuralNetwork_layerData_1_S);
    NeuralNetwork_layerData_2_S = ((NeuralNetwork_layerData_2_W)')*([1;NeuralNetwork_layerData_1_X]);
    NeuralNetwork_layerData_2_X = (NeuralNetwork_layerData_2_S);
    Data_test_label(i) = sign(NeuralNetwork_layerData_2_X);
end
posLabel = find(Data_test_label == 1);
negLabel = find(Data_test_label == -1);
posLabelTrain = find(dataTrainY == 1);
negLabelTrain = find(dataTrainY == -1);
%plot directly
figure(2),
plot(testx(posLabel),testy(posLabel),'y+'),hold on
plot(testx(negLabel),testy(negLabel),'go'),hold on

scatter(dataTrainX(negLabelTrain,1), dataTrainX(negLabelTrain,2), 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(dataTrainX(posLabelTrain,1), dataTrainX(posLabelTrain,2), 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold off

xlabel('Avg. Intensity');
ylabel('Avg. Difference');
legend('y = +1','y = -1','TrainDataset - Not 1','TrainDataset - 1');
title('Neural Network with 10 hidden units');










