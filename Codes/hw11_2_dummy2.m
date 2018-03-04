% function hw11_2_dummy2

errorLOOCV = zeros(numTrainSamples, 1);
for iter_k = numTrainSamples-1:-1:251
    r = 2/sqrt(iter_k);
    error = 0;
    fprintf('Evaluating for k = %d clusters;\n',iter_k);
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
%         tic
        matrixZ = hw11_2_dummy3(data_x, clusterMean, r);
%         t1 = toc;
%         fprintf('\tmatrixZ %f seconds\n',t1);
%         tic
        weight = linearRegression(matrixZ, data_y);
%         t1 = toc;
%         fprintf('\tlinReg %f seconds\n',t1);
%         tic
        test_z = hw11_2_dummy3(test_x, clusterMean, r);
%         t1 = toc;
%         fprintf('\tmatrixZTest %f seconds\n',t1);
%         tic
        error = error + evaluateError(test_z, weight, test_y);
%         t1 = toc;
%         fprintf('\terrorCalc %f seconds\n',t1);
    end
    t1 = toc;
    fprintf('Took %f seconds\n',t1);
    errorLOOCV(iter_k) = error/numTrainSamples;
end


    
    
    
    
    
    
    
    
    
%%
%%