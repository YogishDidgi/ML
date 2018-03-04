% function hw12_q3
dataX = [1 0; -1 0];
dataY = [1; -1];

[numPoints, dataDimension] = size(dataX);

%initialise size
p = zeros(dataDimension + 1, 1);
Q = zeros(dataDimension + 1, dataDimension + 1);
A = zeros(numPoints, dataDimension + 1);
c = ones(numPoints, 1);

%fill the values
Q(2:end, 2:end) = eye(dataDimension);
A(:,1) = dataY;

%A(:,2:end) = (-dataY).*(dataX);
for i = 1:numPoints
    A(i,2:end) = dataY(i)*(dataX(i,:)');
end

%**********Matlab inequality constraint and LFD constraint are opposite*************
A = -A;
c = -c;

%solve
x = quadprog(Q,p,A,c);
bias = x(1);
weight = x(2:end);

%plot figure
xMin = -2;
xMax = 2;
yMin = -2;
yMax = 2;
precision = 0.01;

%varX = xMin:precision:xMax;
%varY = yMin:precision:yMax;
%%
hypothesis = @(x1,x2) weight'*[x1;x2] + bias;
% figure,
% ezplot(hypothesis, [-1 1]);
% hold on;
% scatter(dataX(:,1)',dataX(:,2)');
% hold off;

%%
%z-Transformation

funcZ = @(x1,x2) [power(x1,3) - x2; x1.*x2];
dataZ = (funcZ(dataX(:,1)',dataX(:,2)'))';

A(:,1) = dataY;
for i = 1:numPoints
    A(i,2:end) = dataY(i)*(dataZ(i,:)');
end

%**********Matlab inequality constraint and LFD constraint are opposite*************
A = -A;

%solve
z = quadprog(Q,p,A,c);
biasZ = z(1);
weightZ = z(2:end);
hypothesisZ = @(x1,x2) weightZ'*funcZ(x1,x2) + biasZ;

%%
dataYPosIndex = find(dataY == 1);
dataYNegIndex = find(dataY == -1);
dataXPos = dataX(dataYPosIndex,:);
dataXNeg = dataX(dataYNegIndex,:);
figure(1)
scatter(dataXNeg(:,1), dataXNeg(:,2), 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(dataXPos(:,1), dataXPos(:,2), 'Marker', 'o', 'MarkerEdgeColor', 'b');
hold on
fig1 = ezplot(hypothesis,[-2 2]);
set(fig1,'color','green');
hold on
fig2 = ezplot(hypothesisZ,[-2 2]);
set(fig2,'color','yellow');
hold off
legend('-1 class','+1 class','optimal hyperplane in X-space','optimal hyperplane in Z-space');
xlabel('x1');
ylabel('x2');