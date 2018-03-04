%function hw7_2
%f(x, y) = x.^2 + 2*y.^2 + 2*sin(2*pi*x)*sin(2*pi*y);
%%
%part a
clear all
close all
learningRate1 = 0.01;
learningRate2 = 0.1;
numIters = 50;
xInit = 0.1;
yInit = 0.1;

% v = -2:0.2:2;
% [x,y] = meshgrid(v);
syms x y
z = x.^2 + 2*y.^2 + 2*sin(2*pi*x).*sin(2*pi*y);
gradX = 2*x + 2*2*pi*sin(2*pi*y).*cos(2*pi*x);
gradY = 4*y + 2*2*pi*sin(2*pi*x).*cos(2*pi*y);
grad = [gradX; gradY];

zVal_lr1 = zeros(numIters + 1, 1);
zVal_lr1(1) = double(subs(z, {x, y}, {xInit, yInit}));
zVal_lr2 = zeros(numIters + 1, 1);
zVal_lr2(1) = double(subs(z, {x, y}, {xInit, yInit}));
xPos_lr1 = xInit;
yPos_lr1 = yInit;
xPos_lr2 = xInit;
yPos_lr2 = yInit;

for i = 1:numIters
    gradVal_lr1 = double(subs(grad, {x, y}, {xPos_lr1, yPos_lr1}));
    gradVal_lr2 = double(subs(grad, {x, y}, {xPos_lr2, yPos_lr2}));
    xPos_lr1 = xPos_lr1 - learningRate1*gradVal_lr1(1);
    yPos_lr1 = yPos_lr1 - learningRate1*gradVal_lr1(2);
    xPos_lr2 = xPos_lr2 - learningRate2*gradVal_lr2(1);
    yPos_lr2 = yPos_lr2 - learningRate2*gradVal_lr2(2);
    zVal_lr1(i + 1) = double(subs(z, {x, y}, {xPos_lr1, yPos_lr1}));
    zVal_lr2(i + 1) = double(subs(z, {x, y}, {xPos_lr2, yPos_lr2}));
end
% [px,py] = gradient(z,.2,.2);

iters = 0:1:numIters;
figure,
plot(iters, zVal_lr1, '-r', iters, zVal_lr2, '-b');
xlabel('Iteration');
ylabel('f(x,y)');
legend('Learning rate: 0.01','Learning rate: 0.1');

%%
%part b
clear all
close all
lr = 0.01;
numIters = 50;

xInit = [0.1, 1, -0.5, -1];
yInit = [0.1, 1, -0.5, -1];

syms x y
z = x.^2 + 2*y.^2 + 2*sin(2*pi*x).*sin(2*pi*y);
gradX = 2*x + 2*2*pi*sin(2*pi*y).*cos(2*pi*x);
gradY = 4*y + 2*2*pi*sin(2*pi*x).*cos(2*pi*y);
grad = [gradX; gradY];

minLocX = zeros(length(xInit), 1);
minLocY = zeros(length(xInit), 1);
minVal = zeros(length(xInit), 1);

for i = 1:length(xInit)
    xPos = xInit(i);
    yPos = yInit(i);
    for iter = 1:numIters
        gradVal = double(subs(grad, {x, y}, {xPos, yPos}));
        xPos = xPos - lr*gradVal(1);
        yPos = yPos - lr*gradVal(2);
    end
    minLocX(i) = xPos;
    minLocY(i) = yPos;
    minVal(i) = double(subs(z, {x, y}, {xPos, yPos}));
    fprintf('Init:(%g,%g)\tFinal:(%g,%g)\tMinVal:%g\n', xInit(i), yInit(i), minLocX(i), minLocY(i), minVal(i));
end