function [newW, iter] = PLA(X_Augmented, W, Y)
[numSamples, dim] = size(X_Augmented);
prevW = W;
newW = W;
iter = 0;

while(1)
    flag = false;
    disp(newW);
    disp(iter);
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