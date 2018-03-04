
for iter_k = 251:1:numTrainSamples-1
    r = 2/sqrt(iter_k);
    error = 0;
    fprintf('Evaluating for k = %d clusters;\t',iter_k);
    tic
    for iter_cv = 1:numTrainSamples
        %Leave one out
        data_x = DataTrain_X(1:end ~=iter_cv,:);
        data_y = DataTrain_Y(1:end ~=iter_cv,:);
        test_x = DataTrain_X(iter_cv,:);
        test_y = DataTrain_Y(iter_cv,:);
        %do clustering
        [clusterID, clusterMean] = kmeans(data_x,iter_k);
        %transform to z-space
        matrixZ = hw11_2_computeTransform(data_x, clusterMean, r);
        weight = linearRegression(matrixZ, data_y);
        test_z = hw11_2_computeTransform(test_x, clusterMean, r);
        error = error + evaluateError(test_z, weight, test_y);
    end
    t1 = toc;
    fprintf('Took %f seconds\n',t1);
    errorLOOCV(iter_k) = error/numTrainSamples;
end
[minVal, minIndex] = min(errorLOOCV);
bestK = minIndex;
fprintf('bestK: %d\n',bestK);

figure(1),
plot(kRange,errorLOOCV);
xlabel('K');
ylabel('Ecv');
legend('error-LOOCV');
title('K vs Ecv');
%%