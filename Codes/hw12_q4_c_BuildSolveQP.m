function [N,svAlpha,svX,svY,bias] = hw12_q4_c_BuildSolveQP(numTrainSamples, trainX, trainY,regularizerC)

p = -1*ones(numTrainSamples, 1);
Q = zeros(numTrainSamples, numTrainSamples);
c = zeros(numTrainSamples + 2, 1);
A = zeros(numTrainSamples + 2, numTrainSamples);
A(1,:) = trainY';
A(2,:) = -trainY';
A(3:end,:) = eye(numTrainSamples);
Aeq = zeros(numTrainSamples, numTrainSamples);
Beq = zeros(numTrainSamples, 1);
Aeq = repmat(trainY',numTrainSamples,1);

kernelFunction = @(arg1,arg2) power((1 + arg1'*arg2),8);

for i = 1:numTrainSamples
    for j = i:numTrainSamples
        Q(i,j) = trainY(i)*trainY(j)*kernelFunction(trainX(i,:)',trainX(j,:)');
        Q(j,i) = Q(i,j);
    end
end

A = -A;
c = -c;
% Aeq = [];
% beq = [];
lb = zeros(numTrainSamples, 1);
ub = regularizerC*ones(numTrainSamples, 1);

x = quadprog(Q,p,A,c,[],[],[],ub);
% x = quadprog(Q,p,[],[],Aeq,Beq,lb,ub);

supportVectorsIndex = find((x > lb) & (x < ub));
numSupportVectors = size(supportVectorsIndex,1);
if(numSupportVectors)
    bias = 0;
    for i = 1:numSupportVectors
        bias = bias + trainY(supportVectorsIndex(i))*x(supportVectorsIndex(i))*kernelFunction(trainX(supportVectorsIndex(1),:)',trainX(supportVectorsIndex(i),:)');
    end
    bias = trainY(supportVectorsIndex(1)) - bias;
    
    supportVectorsY = trainY(supportVectorsIndex,:);
    supportVectorsX = trainX(supportVectorsIndex,:);
    supportVectorsAlpha = x(supportVectorsIndex,:);
else
    fprintf('No support vectors found\n');
end

N = numSupportVectors;
svAlpha = supportVectorsAlpha;
svX = supportVectorsX;
svY = supportVectorsY;

