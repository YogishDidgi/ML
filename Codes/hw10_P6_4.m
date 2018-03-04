%hw10_P6_4
clc
% clear all
close all

%%
numSamples = 2000;
rad = 10;
thk = 5;
sep = 5;
rMin = rad;
rMax = rad + thk;

theta = pi*rand(numSamples/2,1);
r = sqrt(rand(numSamples/2,1)*(rMax^2 - rMin^2) + rMin^2);%5*rand(numSamples/2, 1) + rad;
a = 0;
b = 0;
x = a + r.*cos(theta);
y = b + r.*sin(theta);

X_neg = [x y];
X_neg_Augment = [ones(numSamples/2,1) x y];
Y_neg = -1*ones(numSamples/2,1);

theta = pi*rand(numSamples/2,1) + pi;
r = sqrt(rand(numSamples/2,1)*(rMax^2 - rMin^2) + rMin^2);%5*rand(numSamples/2, 1) + rad;
a = rad + thk/2;
b = -sep;
x = a + r.*cos(theta);
y = b + r.*sin(theta);

X_pos = [x y];
X_pos_Augment = [ones(numSamples/2,1) x y];
Y_pos = 1*ones(numSamples/2,1);

X_data = [X_neg; X_pos];
X_data_Augment = [X_neg_Augment; X_pos_Augment];
Y_data = [Y_neg; Y_pos];

xMin = floor(min(X_data(:,1)));
xMax = ceil(max(X_data(:,1)));
yMin = floor(min(X_data(:,2)));
yMax = ceil(max(X_data(:,2)));
precision = 0.1;%0.01;

Data_X = X_data;
Data_Y = Y_data;
[testx,testy] = meshgrid(xMin:precision:xMax,yMin:precision:yMax);
testxCopy = testx;
testyCopy = testy;
testx = reshape(testx,[],1);
testy = reshape(testy,[],1);
Data_test = [testx, testy];

figure,
scatter(X_neg(:,1), X_neg(:,2));
hold on
scatter(X_pos(:,1), X_pos(:,2));
hold off
xlabel('X1');
ylabel('X2');
legend('Neg samples', 'Pos samples');
%%
%1-NN
K = 1;
[idxKnearest,DKnearest] = knnsearch(Data_X,Data_test,'K',1);

idx_Label = zeros(size(idxKnearest));
for i=1:size(idxKnearest,1)
    temp1 = idxKnearest(i,:);
    idx_Label(i,:) = Data_Y(temp1);
end
Label_test = sign(sum(Data_Y(idxKnearest),2));
posLabel = find(Label_test == 1);
negLabel = find(Label_test == -1);

%plot directly
figure(3),
subplot(1,2,1),
plot(testx(posLabel),testy(posLabel),'r+'),hold on
plot(testx(negLabel),testy(negLabel),'bo'),hold on
scatter(X_neg(:,1), X_neg(:,2));hold on
scatter(X_pos(:,1), X_pos(:,2));hold off
xlabel('x1');
ylabel('x2');
legend('y = +1','y = -1');
title('1-NN');
%%
%3-NN
K = 3;
[idxKnearest,DKnearest] = knnsearch(Data_X,Data_test,'K',3);
idx_Label = zeros(size(idxKnearest));
for i=1:size(idxKnearest,1)
    temp1 = idxKnearest(i,:);
    idx_Label(i,:) = Data_Y(temp1);
end
Label_test = sign(sum(Data_Y(idxKnearest),2));
posLabel = find(Label_test == 1);
negLabel = find(Label_test == -1);

%plot directly
figure(3),
subplot(1,2,2)
plot(testx(posLabel),testy(posLabel),'r+'),hold on
plot(testx(negLabel),testy(negLabel),'bo'),hold on
scatter(X_neg(:,1), X_neg(:,2));hold on
scatter(X_pos(:,1), X_pos(:,2));hold off
xlabel('x1');
ylabel('x2');
legend('y = +1','y = -1');
title('3-NN');
%%
if 0
[idxKnearest1,DKnearest1] = knnsearch(Data_X,Data_test,'K',150);%----------K = 150
idx_Label1 = zeros(size(idxKnearest1));
for i=1:size(idxKnearest1,1)
    temp1 = idxKnearest1(i,:);
    idx_Label1(i,:) = Data_Y(temp1);
end
Label_test1 = sign(sum(Data_Y(idxKnearest1),2));
posLabel1 = find(Label_test1 == 1);
negLabel1 = find(Label_test1 == -1);

figure(4),
% subplot(1,2,2)
plot(testx(posLabel),testy(posLabel),'r+'),hold on
plot(testx(negLabel),testy(negLabel),'bo'),hold on
scatter(X_neg(:,1), X_neg(:,2));hold on
scatter(X_pos(:,1), X_pos(:,2));hold off
xlabel('x1');
ylabel('x2');
legend('y = +1','y = -1');
title('3-NN');

[idxKnearest3,DKnearest3] = knnsearch(Data_X,Data_test,'K',3);
idx_Label3 = zeros(size(idxKnearest3));
for i=1:size(idxKnearest3,1)
    temp1 = idxKnearest3(i,:);
    idx_Label3(i,:) = Data_Y(temp1);
end
Label_test3 = sign(sum(Data_Y(idxKnearest3),2));
posLabel3 = find(Label_test3 == 1);
negLabel3 = find(Label_test3 == -1);
end

%%