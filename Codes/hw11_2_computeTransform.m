function matrixZ = hw11_2_computeTransform(data_x, clusterMean, r)
    numPoints = size(data_x,1);
    numClusters = size(clusterMean,1);
    matrixZ = zeros(numPoints, numClusters+1);
    matrixZ(:,1) = 1;
    for i = 1:size(data_x,1)
        z = repmat(data_x(i,:),numClusters,1) - clusterMean;
        z = (sum(z.*z,2))/r;
        z = exp(-(z.*z)/2);
        matrixZ(i,2:numClusters+1) = z';
    end
end
