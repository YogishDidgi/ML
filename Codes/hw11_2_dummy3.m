function matrixZAug = hw11_2_dummy3(data_x, clusterMean, r)
    numPoints = size(data_x,1);
    numClusters = size(clusterMean,1);
    matrixZ = zeros(numPoints, numClusters);
    matrixZAug = zeros(numPoints, numClusters);
    
%     data_x_x = data_x(:,1);
%     data_x_y = data_x(:,2);
%     clusterMean_x = clusterMean(:,1);
%     clusterMean_y = clusterMean(:,2);
%     
%     repmat(data_x_x,
    
    data_x_new = repmat(data_x,[1,1,numClusters]);
    clusterMean_new1 = reshape(clusterMean,[1,2,numClusters]);
    clusterMean_new2 = repmat(clusterMean_new1,[numPoints,1,1]);
    
    matA = data_x_new - clusterMean_new2;
    matB = matA.*matA;
    matC = sum(matB,2)/r;
    matD = exp(-(matC.*matC)/2);
    matrixZ = reshape(matD,[numPoints,numClusters,1]);
%     for i = 1:size(data_x,1)
%         z = repmat(data_x(i,:),numClusters,1) - clusterMean;
%         z = (sum(z.*z,2))/r;
%         z = exp(-(z.*z)/2);
%         matrixZ(i,2:numClusters+1) = z';
%     end
    matrixZAug = [ones(numPoints,1) matrixZ];