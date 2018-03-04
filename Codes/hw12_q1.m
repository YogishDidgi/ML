% function hw12_q1
%
NeuralNetwork = struct('numInputNodes',{},'numOutputNodes',{},'numLayers',{},'layerData',{});
NeuralNetworkLayer = struct('numHiddenNodes',{},'S',{},'X',{},'W',{},'delta',{},'gradient',{});
transformFunction = 'identity';%'identity' or 'tanh'

%%
%Initialization
NeuralNetwork(1).numInputNodes = 2;
NeuralNetwork(1).numOutputNodes = 1;
NeuralNetwork(1).numLayers = 2;%including the output layer
NeuralNetwork(1).layerData = NeuralNetworkLayer;

%Layer 1 data
NeuralNetwork(1).layerData(1).numHiddenNodes = 2;
NeuralNetwork(1).layerData(1).S = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes,1);
NeuralNetwork(1).layerData(1).X = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes,1);%don't include the constant 1 node
NeuralNetwork(1).layerData(1).W = zeros(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);
NeuralNetwork(1).layerData(1).delta = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes,1);%same size as S
NeuralNetwork(1).layerData(1).gradient = zeros(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);

%Layer 2 data
%This is also the output layer
NeuralNetwork(1).layerData(2).numHiddenNodes = 1;
NeuralNetwork(1).layerData(2).S = zeros(NeuralNetwork(1).layerData(2).numHiddenNodes,1);
NeuralNetwork(1).layerData(2).X = zeros(NeuralNetwork(1).layerData(2).numHiddenNodes,1);%don't include the constant 1 node
NeuralNetwork(1).layerData(2).W = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);
NeuralNetwork(1).layerData(2).delta = zeros(NeuralNetwork(1).layerData(2).numHiddenNodes,1);
NeuralNetwork(1).layerData(2).gradient = zeros(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);

%%
%Given data
initWeight = 0.25;
x = [1;1];%[x1 x2]; 2 input vector
y = 1;
NeuralNetwork(1).layerData(1).W = initWeight*ones(NeuralNetwork(1).numInputNodes + 1, NeuralNetwork(1).layerData(1).numHiddenNodes);
NeuralNetwork(1).layerData(2).W = initWeight*ones(NeuralNetwork(1).layerData(1).numHiddenNodes + 1, NeuralNetwork(1).layerData(2).numHiddenNodes);

%%
%Forward propagation
for i = 1:NeuralNetwork(1).numLayers
    if(i == 1)
        %working on first layer
        %input is from input layer
        inputVector = [1; x];
    else
        inputVector = [1; NeuralNetwork(1).layerData(i - 1).X];
    end
    NeuralNetwork(1).layerData(i).S = ((NeuralNetwork(1).layerData(i).W)')*inputVector;
    NeuralNetwork(1).layerData(i).X = hw12_q1_fwdTransformFunction(NeuralNetwork(1).layerData(i).S, transformFunction);
end

%%
%Backpropagation
for i = NeuralNetwork(1).numLayers:-1:1
    if(i == NeuralNetwork(1).numLayers)
        NeuralNetwork(1).layerData(i).delta = 2*(NeuralNetwork(1).layerData(i).X - y)*hw12_q1_bwdTransformFunction(NeuralNetwork(1).layerData(i).X, transformFunction);
    else
        %thetaDash = ones(size(NeuralNetwork(1).layerData(i).X)) - (NeuralNetwork(1).layerData(i).X).*(NeuralNetwork(1).layerData(i).X);
        thetaDash = hw12_q1_bwdTransformFunction(NeuralNetwork(1).layerData(i).X, transformFunction);
        prevDelta = repmat(NeuralNetwork(1).layerData(i + 1).delta', NeuralNetwork(1).layerData(i).numHiddenNodes, 1);
        NeuralNetwork(1).layerData(i).delta = thetaDash.*(NeuralNetwork(1).layerData(i + 1).W(2:end,:).*prevDelta);
    end
end

%%
%Gradient calculation using backpropagation
fprintf('Gradient calculation using backpropagation\n');
for i = 1:NeuralNetwork(1).numLayers
    if(i == 1)
        inputVector = [1; x];
    else
        inputVector = [1; NeuralNetwork(1).layerData(i - 1).X];
    end
    NeuralNetwork(1).layerData(i).gradient = inputVector*(NeuralNetwork(1).layerData(i).delta');
    fprintf('---->Layer %d\n',i);
    disp(NeuralNetwork(1).layerData(i).gradient);
end

%%
%Gradient calculation using perturbation
fprintf('Gradient calculation using perturbation\n');
perturbation = 0.00001;
[NeuralNetwork, error] = hw12_q1_forwardPropagation(NeuralNetwork, transformFunction, x, y);
temp = error;
for i = 1:NeuralNetwork(1).numLayers
    [m,n] = size(NeuralNetwork(1).layerData(i).W);
    gradMatrix = temp*ones(m,n);
    for j = 1:m
        for k = 1:n
            NeuralNetwork(1).layerData(i).W(j,k) = initWeight + perturbation;
            
            [NeuralNetwork, error] = hw12_q1_forwardPropagation(NeuralNetwork, transformFunction, x, y);
            gradMatrix(j,k) = (error - gradMatrix(j,k))/perturbation;
            NeuralNetwork(1).layerData(i).W(j,k) = initWeight;%change it back to original
        end
    end
    fprintf('---->Layer %d\n',i);
    disp(gradMatrix);
end

%%
%Gradient descent
numTrainPoints = 1;
learningRate = 0.01;
stopCriteriaIters = 2*1000000;
stopCriteriaValue = 0.000001;
Ein = 0;
inputDimension = 2;
dataTrainX = zeros(numTrainPoints,inputDimension);
dataTrainY = zeros(numTrainPoints,1);

dataTrainX(1,:) = x';
dataTrainY(1) = y;
iters = 0;
while(1)
    %initialization
    Ein = 0;
    for i = 1:NeuralNetwork(1).numLayers
        NeuralNetwork(1).layerData(i).gradient = zeros(size(NeuralNetwork(1).layerData(i).W));
    end
    %Gradient computation
    for i = 1:numTrainPoints
        [NeuralNetwork, error] = hw12_q1_forwardPropagation(NeuralNetwork, transformFunction, dataTrainX(i,:)', dataTrainY(i));
        Ein = Ein + error;
        NeuralNetwork = hw12_q1_backwardPropagation(NeuralNetwork, transformFunction, dataTrainY(i));
        NeuralNetwork = hw12_q1_gradientCalculation(NeuralNetwork, dataTrainX(i,:)');
    end
    %Normalization of error and accumulated gradients; Also Weights updation
    Ein = Ein/(4*numTrainPoints);
    for i = 1:NeuralNetwork(1).numLayers
        NeuralNetwork(1).layerData(i).gradient = NeuralNetwork(1).layerData(i).gradient/numTrainPoints;
        %Weights updation
        NeuralNetwork(1).layerData(i).W = NeuralNetwork(1).layerData(i).W - learningRate*NeuralNetwork(1).layerData(i).gradient;
    end
    iters = iters + 1;
    if(Ein <= stopCriteriaValue || iters == stopCriteriaIters)
        break;
    end
end
fprintf('After gradient descent: Ein: %f numIters: %d\n', Ein, iters);







