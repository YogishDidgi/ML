%function hw12_q4
load digitsData.mat;
regularizerCMin = 1;
regularizerCMax = 200000;%0.0001;
kernelFunction = @(arg1,arg2) power((1 + arg1'*arg2),8);
p = -1*ones(numTrainSamples, 1);
c = zeros(numTrainSamples + 2, 1);
Q = zeros(numTrainSamples, numTrainSamples);
A = zeros(numTrainSamples + 2, numTrainSamples);
A(1,:) = DataTrain_Y';
A(2,:) = -DataTrain_Y';
A(3:end,:) = eye(numTrainSamples);
for i = 1:numTrainSamples
    for j = 1:numTrainSamples
        Q(i,j) = DataTrain_Y(i)*DataTrain_Y(j)*kernelFunction(DataTrain_X(i,:)',DataTrain_X(j,:)');
%         Q(j,i) = Q(i,j);
    end
end
%**********Matlab inequality constraint and LFD constraint are opposite*************
A = -A;
c = -c;
% Aeq = zeros(numTrainSamples, numTrainSamples);
% for i = 1:numTrainSamples
%     Aeq(i,i) = DataTrain_Y(i);
% end
Aeq = DataTrain_Y';
Beq = zeros(1, 1);
lb = zeros(numTrainSamples, 1);

%%
regularizerC = regularizerCMin;
ub = regularizerC*ones(numTrainSamples, 1);
%solve
tic
%x = quadprog(Q,p,A,c,[],[],lb,ub);
x = quadprog(Q,p,[],[],Aeq,Beq,lb,ub);
t1 = toc;
fprintf('QP solved in %f sec\n',t1);
%%
tic
lbb=(1e-5)*ones(numTrainSamples, 1);
supportVectorsIndex = find((x > lbb) & (x < ub));
numSupportVectors = size(supportVectorsIndex,1);
if(numSupportVectors)
    bias = 0;
    supportVectorsY = DataTrain_Y(supportVectorsIndex,:);
    supportVectorsX = DataTrain_X(supportVectorsIndex,:);
    supportVectorsAlpha = x(supportVectorsIndex,:);
    
    for i = 1:numSupportVectors
        bias = bias + supportVectorsY(i)*supportVectorsAlpha(i)*kernelFunction(supportVectorsX(1,:)',supportVectorsX(i,:)');
    end
    bias = supportVectorsY(1) - bias;
else
    fprintf('No support vectors found\n');
end
t1 = toc;
fprintf('bias found in %f sec\n',t1);
%%
tic
if(numSupportVectors)
    %hypothesis = @(arg1, arg2) sign(((supportVectorsY.*supportVectorsAlpha)')*diag(kernelFunction(supportVectorsX',(repmat([arg1;arg2]',numSupportVectors,1))')) + bias);
    %Plot decision boundary
    xMin = -1;
    xMax = 1;
    yMin = -1;
    yMax = 1;
    precision = 0.05;%0.01;
    [testx,testy] = meshgrid(xMin:precision:xMax,yMin:precision:yMax);
    testx = reshape(testx,[],1);
    testy = reshape(testy,[],1);
    Data_test = [testx, testy];
    label_test = zeros(size(Data_test,1),1);
    fprintf('Beginning of loop\n');
    tic
    prodVector = ((supportVectorsY.*supportVectorsAlpha)');
    for i = 1:size(testx,1)
%         tic
        %label_test(i) = sign(((supportVectorsY.*supportVectorsAlpha)')*diag(kernelFunction(supportVectorsX',(repmat([testx(i) testy(i)],numSupportVectors,1))')) + bias);
        label_test(i) = sign((prodVector*...
                hw12_q4_kernelFunctionVectorOperation(supportVectorsX,repmat([testx(i) testy(i)],numSupportVectors,1),kernelFunction)) + bias);
%         toc
    end
    t1 = toc;
    fprintf('End of loop: %f sec\n',t1);
    posLabel = find(label_test == 1);
    negLabel = find(label_test == -1);
    
    figure,
    plot(testx(posLabel),testy(posLabel),'y+'),hold on
    plot(testx(negLabel),testy(negLabel),'go'),hold on
    scatter(DataTrain_Feature1_Not1, DataTrain_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');hold on
    scatter(DataTrain_Feature1_1, DataTrain_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');hold off
    xlabel('Avg. Intensity');
    ylabel('Avg. Difference');
    legend('y = +1','y = -1','TrainDataset - Not 1','TrainDataset - 1');
    title('SVM with minC');
end
t1=toc;
fprintf('Decision boundary in %f sec\n',t1);

%%
regularizerC = regularizerCMax;
ub = regularizerC*ones(numTrainSamples, 1);
%solve
tic
%x = quadprog(Q,p,A,c,[],[],lb,ub);
x = quadprog(Q,p,[],[],Aeq,Beq,lb,ub);
t1 = toc;
fprintf('QP solved in %f sec\n',t1);
%%
tic
supportVectorsIndex = find((x > lbb) & (x < ub));
numSupportVectors = size(supportVectorsIndex,1);
if(numSupportVectors)
    bias = 0;
    supportVectorsY = DataTrain_Y(supportVectorsIndex,:);
    supportVectorsX = DataTrain_X(supportVectorsIndex,:);
    supportVectorsAlpha = x(supportVectorsIndex,:);
    
    for i = 1:numSupportVectors
        bias = bias + supportVectorsY(i)*supportVectorsAlpha(i)*kernelFunction(supportVectorsX(1,:)',supportVectorsX(i,:)');
    end
    bias = supportVectorsY(1) - bias;
else
    fprintf('No support vectors found\n');
end
t1 = toc;
fprintf('bias found in %f sec\n',t1);
%%
tic
if(numSupportVectors)
    %hypothesis = @(arg1, arg2) sign(((supportVectorsY.*supportVectorsAlpha)')*diag(kernelFunction(supportVectorsX',(repmat([arg1;arg2]',numSupportVectors,1))')) + bias);
    %Plot decision boundary
    xMin = -1;
    xMax = 1;
    yMin = -1;
    yMax = 1;
    precision = 0.05;%0.01;
    [testx,testy] = meshgrid(xMin:precision:xMax,yMin:precision:yMax);
    testx = reshape(testx,[],1);
    testy = reshape(testy,[],1);
    Data_test = [testx, testy];
    label_test = zeros(size(Data_test,1),1);
    fprintf('Beginning of loop\n');
    tic
    prodVector = ((supportVectorsY.*supportVectorsAlpha)');
    for i = 1:size(testx,1)
%         tic
        %label_test(i) = sign(((supportVectorsY.*supportVectorsAlpha)')*diag(kernelFunction(supportVectorsX',(repmat([testx(i) testy(i)],numSupportVectors,1))')) + bias);
        label_test(i) = sign((prodVector*...
                hw12_q4_kernelFunctionVectorOperation(supportVectorsX,repmat([testx(i) testy(i)],numSupportVectors,1),kernelFunction)) + bias);
%         toc
    end
    t1 = toc;
    fprintf('End of loop: %f sec\n',t1);
    posLabel = find(label_test == 1);
    negLabel = find(label_test == -1);
    
    figure,
    plot(testx(posLabel),testy(posLabel),'y+'),hold on
    plot(testx(negLabel),testy(negLabel),'go'),hold on
    scatter(DataTrain_Feature1_Not1, DataTrain_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');hold on
    scatter(DataTrain_Feature1_1, DataTrain_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');hold off
    xlabel('Avg. Intensity');
    ylabel('Avg. Difference');
    legend('y = +1','y = -1','TrainDataset - Not 1','TrainDataset - 1');
    title('SVM with maxC');
end
t1=toc;
fprintf('Decision boundary in %f sec\n',t1);



