% function hw12_q4_c
%digitsDataRead;
load digitsData.mat;

%%
%5-fold cross-validation
cvFold = 10;
numSamplesEachSet = numTrainSamples/cvFold;

samplesSetX = zeros(numSamplesEachSet,size(DataTrain_X,2),cvFold);
samplesSetY = zeros(numSamplesEachSet,size(DataTrain_Y,2),cvFold);

sampleIndex = datasample(1:1:numTrainSamples,numTrainSamples,'Replace',false);
for i = 1:cvFold
    startIndex = (i - 1)*numSamplesEachSet + 1;
    endIndex = startIndex + numSamplesEachSet - 1;
    samplesSetX(:,:,i) = DataTrain_X(sampleIndex(startIndex:endIndex),:);
    samplesSetY(:,:,i) = DataTrain_Y(sampleIndex(startIndex:endIndex),:);
end
kernelFunction = @(arg1,arg2) power((1 + arg1'*arg2),8);

%%
regularizerCMin = 1;
regularizerCMax = 50;
regularizerCPrecision = 5;
regularizerC = regularizerCMin:regularizerCPrecision:regularizerCMax;
Ecv = zeros(size(regularizerC,2),1);

trainX = zeros(numTrainSamples - numSamplesEachSet,size(DataTrain_X,2));
trainY = zeros(numTrainSamples - numSamplesEachSet,size(DataTrain_Y,2));
testX = zeros(numSamplesEachSet,size(DataTrain_X,2));
testY = zeros(numSamplesEachSet,size(DataTrain_Y,2));

for iterC = 1:size(regularizerC,2)
    fprintf('+++++Running regularizer: %f\n',regularizerC(iterC));
    tic
    for iterCV = 1:cvFold
        fprintf('\t\tRunning cvFold: %d\t',iterCV);
        tic
        testX = samplesSetX(:,:,iterCV);
        testY = samplesSetY(:,:,iterCV);
        trainX = reshape(samplesSetX(:,:,1:end ~= iterCV),numTrainSamples - numSamplesEachSet,size(DataTrain_X,2));
        trainY = reshape(samplesSetY(:,:,1:end ~= iterCV),numTrainSamples - numSamplesEachSet,size(DataTrain_Y,2));
        [N,svAlpha,svX,svY,bias] = hw12_q4_c_BuildSolveQP(numTrainSamples - numSamplesEachSet, trainX, trainY, regularizerC(iterC));
        
        prodVector = ((svY.*svAlpha)');
        for i = 1:numSamplesEachSet
            testPointX = testX(i,:);
            testPointY = testY(i,:);
            Ecv(iterC) = Ecv(iterC) + power(sign((prodVector*...
                    hw12_q4_kernelFunctionVectorOperation(svX,repmat([testPointX(1) testPointX(2)],N,1),kernelFunction)) + bias) - testPointY(1),2);
        end
        t1 = toc;
        fprintf('Took: %f sec\n',t1);
    end
    Ecv(iterC) = Ecv(iterC)/numTrainSamples;
    t1 = toc;
    fprintf('-----Took: %f sec\n',t1);
end

%%
[minVal,minIndex] = min(Ecv);

[N,svAlpha,svX,svY,bias] = hw12_q4_c_BuildSolveQP(numTrainSamples, DataTrain_X, DataTrain_Y, regularizerC(minIndex));
if(N)
    xMin = -1;
    xMax = 1;
    yMin = -1;
    yMax = 1;
    precision = 0.1;%0.01;
    [testx,testy] = meshgrid(xMin:precision:xMax,yMin:precision:yMax);
    testx = reshape(testx,[],1);
    testy = reshape(testy,[],1);
    Data_test = [testx, testy];
    label_test = zeros(size(Data_test,1),1);
    fprintf('Beginning of loop\n');
    tic
    prodVector = ((svY.*svAlpha)');
    for i = 1:size(testx,1)
        label_test(i) = sign((prodVector*...
                hw12_q4_kernelFunctionVectorOperation(svX,repmat([testx(i) testy(i)],N,1),kernelFunction)) + bias);
    end
    t1 = toc;
    fprintf('End of loop: %f sec\n',t1);
    posLabel = find(label_test == 1);
    negLabel = find(label_test == -1);
    
    figure,
    %ezplot(hypothesis, [-2 2]);
    plot(testx(posLabel),testy(posLabel),'y+'),hold on
    plot(testx(negLabel),testy(negLabel),'go'),hold on

    scatter(DataTrain_Feature1_Not1, DataTrain_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
    hold on
    scatter(DataTrain_Feature1_1, DataTrain_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
    hold off

    xlabel('Avg. Intensity');
    ylabel('Avg. Difference');
    legend('y = +1','y = -1','TrainDataset - Not 1','TrainDataset - 1');
    title('SVM');
end






