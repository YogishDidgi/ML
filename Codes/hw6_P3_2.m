function hw6_P3_2
clear all
close all
%%
numSamples = 2000;
rad = 10;
thk = 5;

sepArray = 0.2:0.2:5;
numIters = zeros(length(sepArray), 1);
for i = 1:length(sepArray)
sep = sepArray(i);
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

%Part a
W = [0 0 0];
[newW, iter] = PLA(X_data_Augment, W, Y_data);
numIters(i) = iter;
%Part b
% X = X_data_Augment;
% X_T = X_data_Augment';
% X_plus = inv(X_T*X)*X_T;
% newW = X_plus*Y_data;

%Plot

% x = [-(rad+thk):2*(rad+thk)];
% hypothesisFunctionFinal = (-newW(2)/newW(3)).*x + (-newW(1)/newW(3));
% figure,
% scatter(X_neg(:,1), X_neg(:,2));
% hold on
% scatter(X_pos(:,1), X_pos(:,2));
% hold on
% plot(x, hypothesisFunctionFinal, '-g');
% hold off
% hold off
% xlabel('X1');
% ylabel('X2');
% legend('Neg samples', 'Pos samples', 'HypothesisFinal');
% keyboard
end
figure,
plot(sepArray, numIters);
xlabel('sep');
ylabel('num of Iters');


function [newW, iter] = PLA(X_Augmented, W, Y)
[numSamples, dim] = size(X_Augmented);
prevW = W;
newW = W;
iter = 0;

while(1)
    flag = false;
    disp(newW);
    dist(iter);
    for i=1:numSamples
        if(Y(i) == checkSample(X_Augmented(i,:), newW))
            continue;
        else
            flag = true;
            iter = iter + 1;
            newW = prevW + Y(i).*X_Augmented(i,:);
            prevW = newW;
        end
        if(flag == true)
            break;
        end
    end
    if(flag == false)
        break;
    end
end

function output = checkSample(X, W)
output = sign((W*X'));