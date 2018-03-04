% function hw7_1d
%function hw6_digits
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
%%
ATrain = ATrain';
ATest = ATest';

mask_1_Train = find(ATrain(:,1) == 1.0);
mask_5_Train = find(ATrain(:,1) == 5.0);
Data_1_Train = ATrain(mask_1_Train, 2:257);
Data_5_Train = ATrain(mask_5_Train, 2:257);

numSamples_1_Train = size(Data_1_Train, 1);
numSamples_5_Train = size(Data_5_Train, 1);

mask_1_Test = find(ATest(:,1) == 1.0);
mask_5_Test = find(ATest(:,1) == 5.0);
Data_1_Test = ATest(mask_1_Test, 2:257);
Data_5_Test = ATest(mask_5_Test, 2:257);

numSamples_1_Test = size(Data_1_Test, 1);
numSamples_5_Test = size(Data_5_Test, 1);
% sample_1 = Data_1(1, :);
% sample_1 = reshape(sample_1, [16 16])';
% figure,
% subplot(1, 2, 1),imshow(1-sample_1, []);
% 
% sample_5 = Data_5(1, :);
% sample_5 = reshape(sample_5, [16 16])';
% subplot(1, 2, 2),imshow(1-sample_5, []);

intensity_1_Train = sum(Data_1_Train, 2)/256;
intensity_5_Train = sum(Data_5_Train, 2)/256;
intensity_1_Test = sum(Data_1_Test, 2)/256;
intensity_5_Test = sum(Data_5_Test, 2)/256;

Data_1_Train_reshape = reshape(Data_1_Train,[],16,16);
[m n o] = size(Data_1_Train_reshape);
difference_1_Train = zeros(m,1);
for i=1:m
    sample_1(1:16,:) = Data_1_Train_reshape(i,1:16,:);
    sample_1 = sample_1';
%     sample_1_rot = rot90(sample_1,2);
    diff_1 = abs(sample_1 - fliplr(sample_1));
    diff_2 = abs(sample_1 - flipud(sample_1));
%     diff_3 = abs(sample_1 - sample_1_rot);
    difference_1_Train(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

Data_5_Train_reshape = reshape(Data_5_Train,[],16,16);
[m n o] = size(Data_5_Train_reshape);
difference_5_Train = zeros(m,1);
for i=1:m
    sample_5(1:16,:) = Data_5_Train_reshape(i,1:16,:);
    sample_5 = sample_5';
%     sample_5_rot = rot90(sample_5,2);
    diff_1 = abs(sample_5 - fliplr(sample_5));
    diff_2 = abs(sample_5 - flipud(sample_5));
%     diff_3 = abs(sample_5 - sample_5_rot);
    difference_5_Train(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

Data_1_Test_reshape = reshape(Data_1_Test,[],16,16);
[m n o] = size(Data_1_Test_reshape);
difference_1_Test = zeros(m,1);
for i=1:m
    sample_1(1:16,:) = Data_1_Test_reshape(i,1:16,:);
    sample_1 = sample_1';
%     sample_1_rot = rot90(sample_1,2);
    diff_1 = abs(sample_1 - fliplr(sample_1));
    diff_2 = abs(sample_1 - flipud(sample_1));
%     diff_3 = abs(sample_1 - sample_1_rot);
    difference_1_Test(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

Data_5_Test_reshape = reshape(Data_5_Test,[],16,16);
[m n o] = size(Data_5_Test_reshape);
difference_5_Test = zeros(m,1);
for i=1:m
    sample_5(1:16,:) = Data_5_Test_reshape(i,1:16,:);
    sample_5 = sample_5';
%     sample_5_rot = rot90(sample_5,2);
    diff_1 = abs(sample_5 - fliplr(sample_5));
    diff_2 = abs(sample_5 - flipud(sample_5));
%     diff_3 = abs(sample_5 - sample_5_rot);
    difference_5_Test(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

%%
feature_1_Train = [intensity_1_Train; intensity_5_Train];
feature_2_Train = [difference_1_Train; difference_5_Train];
feature_1_Test = [intensity_1_Test; intensity_5_Test];
feature_2_Test = [difference_1_Test; difference_5_Test];


X_data_Train = [1*ones(numSamples_1_Train+numSamples_5_Train, 1) feature_1_Train feature_2_Train];
Y_data_Train = [1*ones(numSamples_1_Train, 1); -1*ones(numSamples_5_Train, 1)];
X0 = X_data_Train(:,1);
X1 = X_data_Train(:,2);
X2 = X_data_Train(:,3);
Z_data_Train = [X0,X1,X2,X1.*X1,X1.*X2,X2.*X2,X1.*(X1.*X1),(X1.*X1).*X2,X1.*(X2.*X2),X2.*(X2.*X2)];
X_data_Test = [1*ones(numSamples_1_Test+numSamples_5_Test, 1) feature_1_Test feature_2_Test];
Y_data_Test = [1*ones(numSamples_1_Test, 1); -1*ones(numSamples_5_Test, 1)];
X0 = X_data_Test(:,1);
X1 = X_data_Test(:,2);
X2 = X_data_Test(:,3);
Z_data_Test = [X0,X1,X2,X1.*X1,X1.*X2,X2.*X2,X1.*(X1.*X1),(X1.*X1).*X2,X1.*(X2.*X2),X2.*(X2.*X2)];
% W = [0 0 0];

Wlin = linearRegression(Z_data_Train, Y_data_Train);
maxIters = 100;
[newW, Ein] = pocketAlgorithmPLA(Z_data_Train, Wlin, Y_data_Train, maxIters);

errorLinReg_Train = evaluateError(Z_data_Train, Wlin, Y_data_Train)
errorPocket_Train = evaluateError(Z_data_Train, newW, Y_data_Train)
errorLinReg_Test = evaluateError(Z_data_Test, Wlin, Y_data_Test)
errorPocket_Test = evaluateError(Z_data_Test, newW, Y_data_Test)

x = [-1:0.01:1];
y = [-1:0.01:1];
[X1,X2] = meshgrid(x,y);
W=Wlin;
hypothesisFunctionLinReg=W(1)+W(2)*X1+W(3)*X2+W(4)*X1.*X1+W(5)*X1.*X2+W(6)*X2.*X2+W(7)*X1.*(X1.*X1)+W(8)*(X1.*X1).*X2+W(9)*X1.*(X2.*X2)+W(10)*X2.*(X2.*X2);
W=newW;
hypothesisFunctionFinal=W(1)+W(2)*X1+W(3)*X2+W(4)*X1.*X1+W(5)*X1.*X2+W(6)*X2.*X2+W(7)*X1.*(X1.*X1)+W(8)*(X1.*X1).*X2+W(9)*X1.*(X2.*X2)+W(10)*X2.*(X2.*X2);
%%
%Plot features
figure,
subplot(1,2,1)
scatter(intensity_5_Train, difference_5_Train, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(intensity_1_Train, difference_1_Train, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
contour(X1,X2,hypothesisFunctionFinal,[0 0],'-g');%plot(x, hypothesisFunctionFinal, '-g');
hold on
contour(X1,X2,hypothesisFunctionLinReg,[0 0],'-y');%plot(x, hypothesisFunctionLinReg, '-y');
hold off
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
legend('Dataset - 5','Dataset - 1','g(x)-PocketPLA','g(x)-LinearRegression');
title('Training Data');

subplot(1,2,2)
scatter(intensity_5_Test, difference_5_Test, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(intensity_1_Test, difference_1_Test, 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
contour(X1,X2,hypothesisFunctionFinal,[0 0],'-g');%plot(x, hypothesisFunctionFinal, '-g');
hold on
contour(X1,X2,hypothesisFunctionLinReg,[0 0],'-y');%plot(x, hypothesisFunctionLinReg, '-y');
hold off
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
legend('Dataset - 5','Dataset - 1','g(x)-PocketPLA','g(x)-LinearRegression');
title('Testing Data');