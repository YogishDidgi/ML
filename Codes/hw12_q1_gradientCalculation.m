function NeuralNetworkOutput = hw12_q1_gradientCalculation(NeuralNetworkInput, inputVector)
%     for i = 1:NeuralNetworkInput(1).numLayers
%         if(i == 1)
%             inputVector = [1; inputVector];
%         else
%             inputVector = [1; NeuralNetworkInput(1).layerData(i - 1).X];
%         end
%         %Accumulation of gradients
%         NeuralNetworkInput(1).layerData(i).gradient = NeuralNetworkInput(1).layerData(i).gradient + inputVector*(NeuralNetworkInput(1).layerData(i).delta');
%     end

%Loop unrolling
    NeuralNetworkInput(1).layerData(1).gradient = NeuralNetworkInput(1).layerData(1).gradient + ([1; inputVector])*(NeuralNetworkInput(1).layerData(1).delta');
    NeuralNetworkInput(1).layerData(2).gradient = NeuralNetworkInput(1).layerData(2).gradient + ([1; NeuralNetworkInput(1).layerData(1).X])*(NeuralNetworkInput(1).layerData(2).delta');
    NeuralNetworkOutput = NeuralNetworkInput;
end