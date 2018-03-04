% function hw10_P6_1
clc
close all
Data = [1 0 -1; 0 1 -1; 0 -1 -1; -1 0 -1; 0 2 1; 0 -2 1; -2 0 1];
Data_X = Data(:,1:2);
Data_Y = Data(:,3);
% xMin = min(X(:,1));
% xMax = max(X(:,1));
% yMin = min(X(:,2));
% yMax = max(X(:,2));
xMin = -6;
xMax = 6;
yMin = -6;
yMax = 6;
precision = 0.1;%0.01;

[testx,testy] = meshgrid(xMin:precision:xMax,yMin:precision:yMax);
testxCopy = testx;
testyCopy = testy;
testx = reshape(testx,[],1);
testy = reshape(testy,[],1);
Data_test = [testx, testy];

D = pdist2(Data_test, Data_X);
[D,idx] = sort(D, 2, 'ascend');
%%
x = Data_X(:,1);
y = Data_X(:,2);
xNeg = x(1:4);
xPos = x(5:7);
yNeg = y(1:4);
yPos = y(5:7);
figure(1),voronoi(x,y);

[vx,vy] = voronoi(x,y);
figure(2),
plot(xNeg,yNeg,'ro',vx,vy,'b-'),hold on
plot(xPos,yPos,'r+',vx,vy,'b-'),hold off
axis equal
xlabel('x1');
ylabel('x2');
title('1-NN');
% xlim([min(X(:,1)) max(X(:,1))])
% ylim([min(X(:,2)) max(X(:,2))])
% [v,c] = voronoin(X);
%%
%1-NN
K = 1;
DKnearest = D(:,1:K);
idxKnearest = idx(:,1:K);
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
plot(testx(negLabel),testy(negLabel),'bo'),hold off
xlabel('x1');
ylabel('x2');
legend('y = +1','y = -1');
title('1-NN');
%%
%3-NN
K = 3;
DKnearest = D(:,1:K);
idxKnearest = idx(:,1:K);
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
plot(testx(negLabel),testy(negLabel),'bo'),hold off
xlabel('x1');
ylabel('x2');
legend('y = +1','y = -1');
title('3-NN');
%%
%Z transformation
Data_Z = Data_X;%Just getting Z to same size as X

Data_Z(:,1) = sqrt(Data_X(:,1).*Data_X(:,1) + Data_X(:,2).*Data_X(:,2));
Data_Z(:,2) = atan(Data_X(:,2)./Data_X(:,1));
x = Data_Z(:,1);
y = Data_Z(:,2);
xNeg = x(1:4);
xPos = x(5:7);
yNeg = y(1:4);
yPos = y(5:7);
%figure(4),voronoi(x,y);

[vx,vy] = voronoi(x,y);
figure(4),
plot(xNeg,yNeg,'ro',vx,vy,'b-'),hold on
plot(xPos,yPos,'r+',vx,vy,'b-'),hold off
axis equal
xlabel('z1');
ylabel('z2');
title('1-NN');

%1-NN
Data_test_Z = Data_test;
Data_test_Z(:,1) = sqrt(Data_test(:,1).*Data_test(:,1) + Data_test(:,2).*Data_test(:,2));
Data_test_Z(:,2) = atan(Data_test(:,2)./Data_test(:,1));
testx = Data_test(:,1);
testy = Data_test(:,2);
D = pdist2(Data_test_Z, Data_Z);
[D,idx] = sort(D, 2, 'ascend');

K = 1;
DKnearest = D(:,1:K);
idxKnearest = idx(:,1:K);
idx_Label = zeros(size(idxKnearest));
for i=1:size(idxKnearest,1)
    temp1 = idxKnearest(i,:);
    idx_Label(i,:) = Data_Y(temp1);
end
Label_test = sign(sum(Data_Y(idxKnearest),2));
posLabel = find(Label_test == 1);
negLabel = find(Label_test == -1);

%plot directly
figure(5),
subplot(1,2,1),
plot(testx(posLabel),testy(posLabel),'r+'),hold on
plot(testx(negLabel),testy(negLabel),'bo'),hold off
xlabel('x1');
ylabel('x2');
legend('y = +1','y = -1');
title('1-NN');

%3-NN
K = 3;
DKnearest = D(:,1:K);
idxKnearest = idx(:,1:K);
idx_Label = zeros(size(idxKnearest));
for i=1:size(idxKnearest,1)
    temp1 = idxKnearest(i,:);
    idx_Label(i,:) = Data_Y(temp1);
end
Label_test = sign(sum(Data_Y(idxKnearest),2));
posLabel = find(Label_test == 1);
negLabel = find(Label_test == -1);

%plot directly
figure(5),
subplot(1,2,2)
plot(testx(posLabel),testy(posLabel),'r+'),hold on
plot(testx(negLabel),testy(negLabel),'bo'),hold off
xlabel('x1');
ylabel('x2');
legend('y = +1','y = -1');
title('3-NN');

