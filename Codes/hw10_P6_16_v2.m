% hw10_P6_16_v2
clc
close all
numTrainPoints = 10000;
numTestPoints = 10000;
numClusters = 10;

trainData = rand(numTrainPoints,2);%[0,1]
% trainData = normrnd(0.5, 0.1, numTrainPoints,2);%[0,1]

testData = -1 + 3*rand(numTestPoints,2);%[-1,2]
nearestNeighbourBruteForce = zeros(numTestPoints,2);
nearestNeighbourBranchBound = zeros(numTestPoints,2);
%%
%Brute force
disp('-Starting Brute force search-');
tic
distArray = zeros(numTrainPoints,1);
for i = 1:numTestPoints
    matA = repmat(testData(i,:),numTrainPoints,1);
    matB = matA - trainData;
    matC = sqrt(sum(matB.*matB,2));
    [minVal,minIndex] = min(matC);
    nearestNeighbourBruteForce(i,:) = trainData(minIndex,:);
%     for j = 1:numTrainPoints
%         matA = (testData(i,:) - trainData(j,:));
%         %matB = matA.*matA;
%         distArray(j) = (sum(matA.*matA,2));%sqrt is removed;
%     end
%     [minValue,minIndex] = min(distArray);
%     nearestNeighbourBruteForce(i,:) = trainData(minIndex,:);
end
t1 = toc;
fprintf('Brute force runs in %f sec\n',t1);
%%
%Partioning data
clusterMeans = zeros(numClusters,2);
clusterMeans(1,:) = trainData(randi(numTrainPoints),:);
disp('-Starting Branch-Bound Partition-');
tic
for i = 2:numClusters
    distArray = zeros(numTrainPoints,i - 1);
    for iter = 1:(i - 1)
        matA = repmat(clusterMeans(iter,:),numTrainPoints,1);
        matB = trainData - matA;
        distArray(:,iter) = sqrt(sum(matB.*matB,2));
    end
    matC = min(distArray,[],2);%Finding closest cluster center for each point
    [maxValue,maxIndex] = max(matC);%Find the point which maximizes the minimum distance
    clusterMeans(i,:) = trainData(maxIndex,:);
end
t2 = toc;
fprintf('Branch-Bound Partition runs in %f sec\n',t2);
%%
figure(1),
subplot(1,2,1),
x = clusterMeans(:,1);
y = clusterMeans(:,2);
[vx,vy] = voronoi(x,y);
plot(trainData(:,1),trainData(:,2),'go');
hold on
plot(x,y,'r+',vx,vy,'b-');
axis equal
xlim([0 1])
ylim([0 1])
xlabel('x');
ylabel('y');
title('Initial cluster centers');
%%
disp('-Starting Branch-Bound bookkeeping-');
tic
%Find label of each point
labelMatTrain = zeros(numTrainPoints,1);
for i = 1:numTrainPoints
    matA = repmat(trainData(i,:),numClusters,1);
    matB = matA - clusterMeans;
    matC = sqrt(sum(matB.*matB,2));
    [minVal,minIndex] = min(matC);
    labelMatTrain(i) = minIndex;
end

%Updating cluster centers
clusterMeansNew = zeros(size(clusterMeans));
for i = 1:numClusters
    clusterPointsIndex = find(labelMatTrain == i);
    matA = trainData(clusterPointsIndex,:);
    clusterMeansNew(i,:) = sum(matA,1)/size(clusterPointsIndex,1);
end

%Creating matrices for each of the partition points
clusterData = cell(numClusters,1);
for i = 1:numClusters
    labelI = find(labelMatTrain == i);
    clusterData{i} = num2cell(trainData(labelI,:));
end

t3 = toc;
fprintf('Branch-Bound bookkeeping runs in %f sec\n',t3);
%%
figure(1),
subplot(1,2,2),
x = clusterMeansNew(:,1);
y = clusterMeansNew(:,2);
[vx,vy] = voronoi(x,y);
plot(trainData(:,1),trainData(:,2),'go');
hold on
plot(x,y,'r+',vx,vy,'b-');
hold off
axis equal
xlim([0 1])
ylim([0 1])
xlabel('x');
ylabel('y');
legend('Training data','Cluster centers','Cluster boundaries');
title('Recalculating cluster centers');
%%
%Searching nearest neighbour using Branch & Bound
disp('-Starting Branch-Bound nearest neighbour search-');
tic
for i = 1:numTestPoints
    matA = repmat(testData(i,:),numClusters,1);
    matB = matA - clusterMeansNew;
    matC = sqrt(sum(matB.*matB,2));
    [minVal,minIndex] = min(matC);%Finds nearest cluster
    if 0
        matD = find(labelMatTrain == minIndex);%All points (indices of those points) in that cluster
        matE = repmat(testData(i,:),size(matD,1),1);
        matF = trainData(matD,:);
        matG = matE - matF;
        matH = sqrt(sum(matG.*matG,2));
        [minVal,minIndex] = min(matH);%Finds nearest point in that cluster
        nearestNeighbourBranchBound(i,:) = matF(minIndex);
    else
        matD = cell2mat(clusterData{minIndex});
        matE = repmat(testData(i,:),size(matD,1),1);
        matF = matE - matD;
        matG = sqrt(sum(matF.*matF,2));
        [minVal,minIndex] = min(matG);%Finds best match amongst the cluster
        nearestNeighbourBranchBound(i,:) = matD(minIndex);
    end
end
t4 = toc;
fprintf('Branch-Bound nearest neighbour search runs in %f sec\n',t4);



