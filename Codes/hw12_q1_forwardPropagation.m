function [NeuralNetworkOutput, error] = hw12_q1_forwardPropagation(NeuralNetworkInput, transformFunction, inputVector, outputVector)
%     for i = 1:NeuralNetworkInput(1).numLayers
%         if(i == 1)
%             %working on first layer
%             %input is from input layer
%             inputVector = [1; inputVector];
%         else
%             inputVector = [1; NeuralNetworkInput(1).layerData(i - 1).X];
%         end
%         NeuralNetworkInput(1).layerData(i).S = ((NeuralNetworkInput(1).layerData(i).W)')*inputVector;
%         %NeuralNetworkInput(1).layerData(i).X = hw12_q1_fwdTransformFunction(NeuralNetworkInput(1).layerData(i).S, transformFunction);
%         NeuralNetworkInput(1).layerData(i).X = NeuralNetworkInput(1).layerData(i).S;
%     end

%Loop unrolling
    NeuralNetworkInput(1).layerData(1).S = ((NeuralNetworkInput(1).layerData(1).W)')*([1;inputVector]);
    NeuralNetworkInput(1).layerData(1).X = NeuralNetworkInput(1).layerData(1).S;
    
    NeuralNetworkInput(1).layerData(2).S = ((NeuralNetworkInput(1).layerData(2).W)')*([1;NeuralNetworkInput(1).layerData(1).X]);
    NeuralNetworkInput(1).layerData(2).X = NeuralNetworkInput(1).layerData(2).S;

    NeuralNetworkOutput = NeuralNetworkInput;
    error = (NeuralNetworkInput(1).layerData(NeuralNetworkInput(1).numLayers).X - outputVector);
    error = error.*error;
end