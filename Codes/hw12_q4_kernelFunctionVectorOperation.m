function output = hw12_q4_kernelFunctionVectorOperation(input1, input2, kernelFunction)
output = zeros(size(input1,1),1);
for i = 1:size(input1,1)
    output(i) = kernelFunction(input1(i,:)',input2(i,:)');
end
end