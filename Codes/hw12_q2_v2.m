%function hw12_q2
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
NeuralNetwork = struct('numInputNodes',{},'numOutputNodes',{},'numLayers',{},'layerData',{});
NeuralNetworkLayer = struct('numHiddenNodes',{},'S',{},'X',{},'W',{},'delta',{},'gradient',{},'Wprev',{});
transformFunction = 'identity';%'identity' or 'tanh'

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
randMin = 0.01;
randMax = 0.2;
NeuralNetwork(1).layerData(1).W = randMin + (randMax - randMin)*rand(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);
NeuralNetwork(1).layerData(2).W = randMin + (randMax - randMin)*rand(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);

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
learningRateBeta = 0.5;
stopCriteriaIters = 2*1000000;
Ein = 0;
Ein_prev = realmax;
iters = 0;
dataTrainX = DataTrain_X;
dataTrainY = DataTrain_Y;
numTrainPoints = numTrainSamples;
while(1)
    tic
    %initialization
    Ein = 0;
    NeuralNetwork(1).layerData(1).gradient = zeros(size(NeuralNetwork(1).layerData(1).W));
    NeuralNetwork(1).layerData(2).gradient = zeros(size(NeuralNetwork(1).layerData(2).W));
    %Gradient computation
    for i = 1:numTrainPoints
        %Forward prop
        NeuralNetwork(1).layerData(1).S = ((NeuralNetwork(1).layerData(1).W)')*([1;dataTrainX(i,:)']);
        NeuralNetwork(1).layerData(1).X = NeuralNetwork(1).layerData(1).S;
        NeuralNetwork(1).layerData(2).S = ((NeuralNetwork(1).layerData(2).W)')*([1;NeuralNetwork(1).layerData(1).X]);
        NeuralNetwork(1).layerData(2).X = NeuralNetwork(1).layerData(2).S;
        error = (NeuralNetwork(1).layerData(NeuralNetwork(1).numLayers).X - dataTrainY(i));
        error = error.*error;
        
        Ein = Ein + error;
        %Backward prop
        NeuralNetwork(1).layerData(2).delta = 2*(NeuralNetwork(1).layerData(2).X - dataTrainY(i))*(ones(size(NeuralNetwork(1).layerData(2).X)));
        thetaDash = ones(size(NeuralNetwork(1).layerData(1).X));
        prevDelta = repmat(NeuralNetwork(1).layerData(2).delta', NeuralNetwork(1).layerData(1).numHiddenNodes, 1);
        NeuralNetwork(1).layerData(1).delta = thetaDash.*(NeuralNetwork(1).layerData(2).W(2:end,:).*prevDelta);
        
        %Gradient calc
        NeuralNetwork(1).layerData(1).gradient = NeuralNetwork(1).layerData(1).gradient + ([1; dataTrainX(i,:)'])*(NeuralNetwork(1).layerData(1).delta');
        NeuralNetwork(1).layerData(2).gradient = NeuralNetwork(1).layerData(2).gradient + ([1; NeuralNetwork(1).layerData(1).X])*(NeuralNetwork(1).layerData(2).delta');
    end
    %Normalization of error and accumulated gradients; Also Weights updation
    Ein = Ein/(4*numTrainPoints);
    
    if(Ein < Ein_prev)
        %Accept the update
        learningRatePrev = learningRate;
        learningRate = learningRateAlpha*learningRate;
        
        NeuralNetwork(1).layerData(1).gradient = NeuralNetwork(1).layerData(1).gradient/numTrainPoints;
        %Weights storing and updation
        NeuralNetwork(1).layerData(1).Wprev = NeuralNetwork(1).layerData(1).W;
        NeuralNetwork(1).layerData(1).W = NeuralNetwork(1).layerData(1).W - learningRate*NeuralNetwork(1).layerData(1).gradient;
        
        NeuralNetwork(1).layerData(2).gradient = NeuralNetwork(1).layerData(2).gradient/numTrainPoints;
        %Weights storing and updation
        NeuralNetwork(1).layerData(2).Wprev = NeuralNetwork(1).layerData(2).W;
        NeuralNetwork(1).layerData(2).W = NeuralNetwork(1).layerData(2).W - learningRate*NeuralNetwork(1).layerData(2).gradient;
    else
        %Reject the update
        learningRate = learningRateBeta*learningRatePrev;
        learningRatePrev = learningRate;
        
        NeuralNetwork(1).layerData(1).gradient = NeuralNetwork(1).layerData(1).gradient/numTrainPoints;
        %Weights storing and updation
        NeuralNetwork(1).layerData(1).W = NeuralNetwork(1).layerData(1).Wprev - learningRate*NeuralNetwork(1).layerData(1).gradient;
        NeuralNetwork(1).layerData(1).Wprev = NeuralNetwork(1).layerData(1).W;
       
        NeuralNetwork(1).layerData(2).gradient = NeuralNetwork(1).layerData(2).gradient/numTrainPoints;
        %Weights storing and updation
        NeuralNetwork(1).layerData(2).W = NeuralNetwork(1).layerData(2).Wprev - learningRate*NeuralNetwork(1).layerData(2).gradient;
        NeuralNetwork(1).layerData(2).Wprev = NeuralNetwork(1).layerData(2).W;
    end
    Ein_prev = Ein;
    iters = iters + 1;
    t1 = toc;
    fprintf('Iter: %d took %f sec\n', iters, t1);
    if(iters == stopCriteriaIters)
        break;
    end
end
fprintf('After gradient descent: Ein: %f numIters: %d\n', Ein, iters);















