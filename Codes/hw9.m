% function hw9
clear all
close all

fileIDTrain = fopen('ZipDigits.train','r');
fileIDTest = fopen('ZipDigits.test','r');
formatSpec = '%f';
sizeATrain = [257, Inf];
sizeATest = [257, Inf];
ATrain = fscanf(fileIDTrain,formatSpec,sizeATrain);
ATest = fscanf(fileIDTest,formatSpec,sizeATest);
fclose(fileIDTrain);
fclose(fileIDTest);
ATrain = ATrain';
ATest = ATest';
disp('Finished reading data');
%%
if 1
    syms x1 x2
    L0=ones(size(x1,1),1);
    L1_x1=legendreP(1,x1);L1_x2=legendreP(1,x2);
    L2_x1=legendreP(2,x1);L2_x2=legendreP(2,x2);
    L3_x1=legendreP(3,x1);L3_x2=legendreP(3,x2);
    L4_x1=legendreP(4,x1);L4_x2=legendreP(4,x2);
    L5_x1=legendreP(5,x1);L5_x2=legendreP(5,x2);
    L6_x1=legendreP(6,x1);L6_x2=legendreP(6,x2);
    L7_x1=legendreP(7,x1);L7_x2=legendreP(7,x2);
    L8_x1=legendreP(8,x1);L8_x2=legendreP(8,x2);

    L1 = [L1_x1,L1_x2];
    L2 = [L2_x1,L1_x1.*L1_x2,L2_x2];
    L3 = [L3_x1,L2_x1.*L1_x2,L1_x1.*L2_x2,L3_x2];
    L4 = [L4_x1,L3_x1.*L1_x2,L2_x1.*L2_x2,L1_x1.*L3_x2,L4_x2];
    L5 = [L5_x1,L4_x1.*L1_x2,L3_x1.*L2_x2,L2_x1.*L3_x2,L1_x1.*L4_x2,L5_x2];
    L6 = [L6_x1,L5_x1.*L1_x2,L4_x1.*L2_x2,L3_x1.*L3_x2,L2_x1.*L4_x2,L1_x1.*L5_x2,L6_x2];
    L7 = [L7_x1,L6_x1.*L1_x2,L5_x1.*L2_x2,L4_x1.*L3_x2,L3_x1.*L4_x2,L2_x1.*L5_x2,L1_x1.*L6_x2,L7_x2];
    L8 = [L8_x1,L7_x1.*L1_x2,L6_x1.*L2_x2,L5_x1.*L3_x2,L4_x1.*L4_x2,L3_x1.*L5_x2,L2_x1.*L6_x2,L1_x1.*L7_x2,L8_x2];

    Z=[L0,L1,L2,L3,L4,L5,L6,L7,L8];

%     Z_FunctionHandle = matlabFunction(Z,'file','PolyTransform','Vars',{x1, x2},'Outputs',{'Z_Value'});
%     disp('Finished poly function creation; Edit 1st feature transform manually');
end
%%
DataSet = [ATrain; ATest];
DataSet_y = -1*ones(size(DataSet,1),1);
Mask_1 = find(DataSet(:,1) == 1.0);
DataSet_y(Mask_1) = 1.0;
DataSet_Unlabelled = DataSet(:,2:257);
DataSet_Feature1 = sum(DataSet_Unlabelled, 2)/256;

DataSet_Unlabelled_Reshape = reshape(DataSet_Unlabelled,[],16,16);
[m n o] = size(DataSet_Unlabelled_Reshape);
DataSet_Feature2 = zeros(m,1);
for i=1:m
    sample_1(1:16,:) = DataSet_Unlabelled_Reshape(i,1:16,:);
    sample_1 = sample_1';
%     sample_1_rot = rot90(sample_1,2);
    diff_1 = abs(sample_1 - fliplr(sample_1));
    diff_2 = abs(sample_1 - flipud(sample_1));
%     diff_3 = abs(sample_1 - sample_1_rot);
    DataSet_Feature2(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

DataSet_Feature1_Normalised = mapminmax(DataSet_Feature1',-1,1)';
DataSet_Feature2_Normalised = mapminmax(DataSet_Feature2',-1,1)';
disp('Finished pre-processing data');
%%
numTrainSamples = 300;
numTestSamples = m - numTrainSamples;
[DataTrain_Feature1, DataTrainMask] = datasample(DataSet_Feature1_Normalised,numTrainSamples,1,'Replace',false);
DataTrainMask = DataTrainMask';
%DataTrainMask = randi(size(DataSet,1),numTrainSamples,1);
%DataTrain_Feature1 = DataSet_Feature1(DataTrainMask);
DataTrain_Feature2 = DataSet_Feature2_Normalised(DataTrainMask);
DataTrain_y = DataSet_y(DataTrainMask);

DummyMask = ones(size(DataSet,1),1);
DummyMask(DataTrainMask) = 0;
DataTest_Feature1 = DataSet_Feature1_Normalised(find(DummyMask));
DataTest_Feature2 = DataSet_Feature2_Normalised(find(DummyMask));
DataTest_y = DataSet_y(find(DummyMask));
disp('Done with data splitting');

%%
DataTrain_Z = PolyTransform(DataTrain_Feature1,DataTrain_Feature2);
DataTest_Z = PolyTransform(DataTest_Feature1,DataTest_Feature2);
% tic
% DataTrain_Z = subs(Z, {x1,x2}, {DataTrain_Feature1,DataTrain_Feature2});
% toc
% DataTrain_Z_double = double(DataTrain_Z);
disp('ZTrain & ZTest calculation completed');
%%
% syms X1 X2
% Z1 = [1,X1,X2,X1.*X1,X1.*X2,X2.*X2,X1.*(X1.*X1),(X1.*X1).*X2,X1.*(X2.*X2),X2.*(X2.*X2)];
% Data_Z1 = subs(Z1, {X1,X2}, {DataTrain_Feature1,DataTrain_Feature2});
% disp('Z1 calculation completed');
% Wlin1 = linearRegression(double(Data_Z1), DataTrain_y);
% size(Wlin1)

%%

Wlin_0 = linearRegression(DataTrain_Z, DataTrain_y);
errorLinReg_Train_0 = evaluateError(DataTrain_Z, Wlin_0, DataTrain_y);
hypothesisFunctionLinReg_0 = Z*Wlin_0;

Wlin_2 = linearRegression(DataTrain_Z, DataTrain_y, 2);
errorLinReg_Train_2 = evaluateError(DataTrain_Z, Wlin_2, DataTrain_y);
hypothesisFunctionLinReg_2 = Z*Wlin_2;
disp('Wlin calculation completed');
%%
DataTrain_1_Mask = find(DataTrain_y == 1.0);
DataTrain_Not1_Mask = find(DataTrain_y == -1.0);
DataTrain_Feature1_1 = DataTrain_Feature1(DataTrain_1_Mask);
DataTrain_Feature1_Not1 = DataTrain_Feature1(DataTrain_Not1_Mask);
DataTrain_Feature2_1 = DataTrain_Feature2(DataTrain_1_Mask);
DataTrain_Feature2_Not1 = DataTrain_Feature2(DataTrain_Not1_Mask);

DataTest_1_Mask = find(DataTest_y == 1.0);
DataTest_Not1_Mask = find(DataTest_y == -1.0);
DataTest_Feature1_1 = DataTest_Feature1(DataTest_1_Mask);
DataTest_Feature1_Not1 = DataTest_Feature1(DataTest_Not1_Mask);
DataTest_Feature2_1 = DataTest_Feature2(DataTest_1_Mask);
DataTest_Feature2_Not1 = DataTest_Feature2(DataTest_Not1_Mask);

%Plot features
figure(1),
subplot(1,3,1)
scatter(DataTrain_Feature1_Not1, DataTrain_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTrain_Feature1_1, DataTrain_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
ezplot(hypothesisFunctionLinReg_0,[-1 1]);
hold off
legend('Dataset - Not 1','Dataset - 1','g(x)-lambda=0');
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
title('Training Data');
subplot(1,3,2)
scatter(DataTrain_Feature1_Not1, DataTrain_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTrain_Feature1_1, DataTrain_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
ezplot(hypothesisFunctionLinReg_2,[-1 1]);
hold off
legend('Dataset - Not 1','Dataset - 1','g(x)-lambda=2');
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
title('Training Data');

%%
%Cross validation
lambda=0:0.01:2;
Ecv = zeros(length(lambda),1);
Mat_Z_Train = DataTrain_Z;
Mat_ZT_Train = DataTrain_Z';
Mat_ZT_MatZ_Train = Mat_ZT_Train*Mat_Z_Train;

% tic
% DataTest_Z = subs(Z, {x1,x2}, {DataTest_Feature1,DataTest_Feature2});
% toc
% tic
% DataTest_Z_double = double(DataTest_Z);
% toc
% disp('ZTrain calculation completed');
%%
Mat_Z_Test = DataTest_Z;
Mat_ZT_Test = DataTest_Z';
Mat_ZT_MatZ_Test = Mat_ZT_Test*Mat_Z_Test;
%%
for i=1:length(lambda)
    term1 = pinv(Mat_ZT_MatZ_Train + lambda(i)*eye(size(Mat_ZT_MatZ_Train)))*Mat_ZT_Train;
    H_Train = Mat_Z_Train*term1;
    yCap_Train = H_Train*DataTrain_y;%-----------------------------sign()???
    tempSum_Train = 0;
    for j = 1:numTrainSamples
        tempSum_Train = tempSum_Train + ((yCap_Train(j) - DataTrain_y(j))/(1 - H_Train(j,j))).^2;
    end
    
    Weight = term1*DataTrain_y;
    yCap_Test = Mat_Z_Test*Weight;%-----------------------------sign()???
    tempSum_Test = 0;
    for j = 1:numTestSamples
        tempSum_Test = tempSum_Test + (yCap_Test(j) - DataTest_y(j)).^2;
    end
    
    Ecv(i) = (1/numTrainSamples)*(tempSum_Train);
    Etest(i) = (1/numTestSamples)*(tempSum_Test);
end
disp('Finished calculation of Ecv and Etest');
figure(2),
plot(lambda, Ecv, '-r');hold on;
plot(lambda, Etest, '-g');
hold off
legend('Ecv','Etest');
xlabel('lambda');
ylabel('Error');
% title('Training Data');

%%
% Zfunc = matlabFunction(Zcopy,'file','ZFunction','Vars',{x3, x4},'Outputs',{'out_Z'});
% out = Zfunc(DataTrain_Feature1,DataTrain_Feature2);
% W_0 = linearRegression(out, DataTrain_y);
% err1 = evaluateError(out, W_0, DataTrain_y);

%%
[minEcv_Value, minEcv_Index] = min(Ecv);
lambda_best = lambda(minEcv_Index)
term1 = pinv(Mat_ZT_MatZ_Train + lambda_best*eye(size(Mat_ZT_MatZ_Train)))*Mat_ZT_Train;
weight_best = term1*DataTrain_y;

hypothesisFunctionLinReg_best = Z*weight_best;
figure(1),
subplot(1,3,3),
scatter(DataTrain_Feature1_Not1, DataTrain_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTrain_Feature1_1, DataTrain_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
ezplot(hypothesisFunctionLinReg_best,[-1 1]);
hold off
legend('Dataset - Not 1','Dataset - 1','g(x)-lambda=best');
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
title('Training Data');
%%
figure(3),
subplot(1,3,1)
scatter(DataTest_Feature1_Not1, DataTest_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTest_Feature1_1, DataTest_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
ezplot(hypothesisFunctionLinReg_0,[-1 1]);
hold off
legend('Dataset - Not 1','Dataset - 1','g(x)-lambda=0');
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
title('Testing Data');
subplot(1,3,2)
scatter(DataTest_Feature1_Not1, DataTest_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTest_Feature1_1, DataTest_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
ezplot(hypothesisFunctionLinReg_2,[-1 1]);
hold off
legend('Dataset - Not 1','Dataset - 1','g(x)-lambda=2');
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
title('Testing Data');
subplot(1,3,3)
scatter(DataTest_Feature1_Not1, DataTest_Feature2_Not1, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(DataTest_Feature1_1, DataTest_Feature2_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
ezplot(hypothesisFunctionLinReg_best,[-1 1]);
hold off
legend('Dataset - Not 1','Dataset - 1','g(x)-lambda=best');
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
title('Testing Data');