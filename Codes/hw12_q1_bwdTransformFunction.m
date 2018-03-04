function output = hw12_q1_bwdTransformFunction(input, transformFunction)
    if(strcmp(transformFunction,'identity'))
        output = ones(size(input));
    else
        output = ones(size(input)) - input.*input;
    end
end