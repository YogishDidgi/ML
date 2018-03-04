function output = hw12_q1_fwdTransformFunction(input, transformFunction)
    if(strcmp(transformFunction,'identity'))
        output = input;
    else
        output = tanh(input);
    end
end