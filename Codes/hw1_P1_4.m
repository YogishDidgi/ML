%PLA implementation
function [W, iter] = main(numSamples, dim)
    close all;
    X = generateData(numSamples, dim);
    x = [-5:5];
    targetFunction = -x;
%     disp('X is : ');
%     disp(X);
    Y(1:numSamples/2) = 1;
    Y(numSamples/2 + 1:numSamples) = -1;
%     disp('Y is : ');
%     disp(Y);
    
    figure, plot(X(1:numSamples/2, 1), X(1:numSamples/2, 2), '+', X(numSamples/2 + 1:numSamples, 1), X(numSamples/2 + 1:numSamples, 2), 'o', x, targetFunction, '-r');
    xlabel('X1');
    ylabel('X2');
    legend('+1 samples', '-1 samples', 'Target: x+y=0');
    
    
    X_Augmented(1:numSamples, 1) = 1;
    X_Augmented(1:numSamples, 2:(dim+1)) = X;
%     disp('X_Augmented is : ');
%     disp(X_Augmented);
%     W = zeros(1, dim + 1);
    W = [0 -1 1];
    hypothesisFunctionInitial = (-W(2)/W(3)).*x + (-W(1)/W(3));
    figure, plot(X(1:numSamples/2, 1), X(1:numSamples/2, 2), '+', X(numSamples/2 + 1:numSamples, 1), X(numSamples/2 + 1:numSamples, 2), 'o', x, targetFunction, '-r');
    hold on;
    plot(x, hypothesisFunctionInitial, '-y');
    hold on;
%     W = [0 1 1];
    [newW, iter] = PLA(X_Augmented, W, Y)
    W = newW;
    hypothesisFunctionFinal = (-W(2)/W(3)).*x + (-W(1)/W(3));
    plot(x, hypothesisFunctionFinal, '-g');
    xlabel('X1');
    ylabel('X2');
    legend('+1 samples', '-1 samples', 'Target: x+y=0', 'HypothesisInitial', 'HypothesisFinal');
end

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
end

function output = checkSample(X, W)
    output = sign((W*X'));
end

%Generation of random sample data
function X = generateData(numSamples, dim)
    X(1:numSamples/2, :) = (5)*rand(numSamples/2, dim);
    X(numSamples/2 + 1:numSamples, :) = (-5)*rand(numSamples - numSamples/2, dim);
end
