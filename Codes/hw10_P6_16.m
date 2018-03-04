% hw10_P6_16
clc
close all
numTrainPoints = 10000;
numClasses = 10;

trainData = rand(numTrainPoints,2);

figure,
scatter(trainData(:,1),trainData(:,2));
xlim([-1 2])
ylim([-1 2])

%%
%Generating partitions
clusterMeans = zeros(numClasses,2);
if 0
    pairWiseDist = pdist(trainData);
    pairWiseDistMat = squareform(pairWiseDist);
end
%First center randomly chosen
clusterMeansIndex = randi(numTrainPoints);
clusterMeans(1,:) = trainData(clusterMeansIndex,:);

for i = 2:numClasses
    %Find point which is at a maximum distance from all the current means
    if 0
        distSum = zeros(numPoints,1);
        for j = 1:numPoints
           for iter = 1:(i - 1)
                clusterCenterIndex = find(ismember(trainData,clusterMeans(iter,:)),1);
                distSum(j) = distSum(j) + pairWiseDistMat(clusterCenterIndex,j);
           end
        end
        [maxSum,maxDistIndex] = max(distSum);
        clusterMeans(i,:) = trainData(maxDistIndex,:);    
    else
        if 0
            %Find next center based on max of sum of dist to current
            %centers
            distSum = zeros(numPoints,1);
            distMat = zeros(numPoints,i - 1);
            for iter = 1:(i - 1)
                matA = repmat(clusterMeans(iter,:),numPoints,1);
                matB = trainData - matA;
                distMat(:,iter) = sqrt(sum(matB.*matB,2));
            end
            distSum = sum(distMat,2);
            [maxSum,maxDistIndex] = max(distSum);
            clusterMeans(i,:) = trainData(maxDistIndex,:);
        else
            %Find next center based on max of min of dist to each current
            %center.
            %The distance of a point from a set is the distance to 
            %its nearest neighbor in the set.
            distMat = zeros(numTrainPoints,i - 1);
            for iter = 1:(i - 1)
                matA = repmat(clusterMeans(iter,:),numTrainPoints,1);
                matB = trainData - matA;
                distMat(:,iter) = sqrt(sum(matB.*matB,2));
            end
            matC = min(distMat,[],2);
            [maxValue,maxIndex] = max(matC);
            clusterMeans(i,:) = trainData(maxIndex,:);
        end
    end
    fprintf('Finished cluster %d\n',i);
end
fprintf('Done with cluster calculations\n');

figure,
voronoi(clusterMeans(:,1),clusterMeans(:,2));

%%
%Find label of each point
labelMatTrain = zeros(numTrainPoints,1);
for i = 1:numTrainPoints
    matA = repmat(trainData(i,:),numClasses,1);
    matB = matA - clusterMeans;
    matC = sqrt(sum(matB.*matB,2));
    [minVal,minIndex] = min(matC);
    labelMatTrain(i) = minIndex;
end

clusterMeansNew = zeros(size(clusterMeans));
%Get centroid of clusters
for i = 1:numClasses
    clusterPointsIndex = find(labelMatTrain == i);
    matA = trainData(clusterPointsIndex,:);
    clusterMeansNew(i,:) = sum(matA,1)/size(clusterPointsIndex);
end

figure,
voronoi(clusterMeansNew(:,1),clusterMeansNew(:,2));
%%
%Generate query points
numTestPoints = 10000;
testData = -1 + 3*rand(numTestPoints,2);


figure,
scatter(testData(:,1),testData(:,2));
xlim([-2 3])
ylim([-2 3])

labelMatTest = zeros(numTestPoints,1);
%Brute force
tic
for i = 1:numTestPoints
    matA = repmat(testData(i,:),numTrainPoints,1);
    matB = matA - trainData;
    matC = sqrt(sum(matB.*matB,2));
    [minVal,minIndex] = min(matC);
    labelMatTest(i) = minIndex;%labelMatTrain(minIndex);
end
t2 = toc;
fprintf('Brute force time: %f sec\n',t2);

labelMatTestBB = zeros(numTestPoints,1);
clusterData = cell(numClasses,1);
for i = 1:numClasses
    labelI = find(labelMatTrain == i);
    clusterData{i} = num2cell(trainData(labelI,:));
end
tic
%Branch and Bound
for i = 1:numTestPoints
    matA = repmat(testData(i,:),numClasses,1);
    matB = matA - clusterMeansNew;
    matC = sqrt(sum(matB.*matB,2));
    [minVal,minIndex] = min(matC);%Finds best label
    matD = cell2mat(clusterData{minIndex});
    matE = repmat(testData(i,:),size(matD,1),1);
    matF = matE - matD;
    matG = sqrt(sum(matF.*matF,2));
    [minVal,minIndex] = min(matG);%Finds best match amongst the cluster
    labelMatTestBB(i) = find(ismember(trainData,matD(minIndex)),1);
end
t2 = toc;
fprintf('Branch and Bound time: %f sec\n',t2);
difference = sum(abs(labelMatTestBB - labelMatTest))


