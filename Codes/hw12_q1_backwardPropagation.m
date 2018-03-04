function NeuralNetworkOutput = hw12_q1_backwardPropagation(NeuralNetworkInput, transformFunction, outputVector)
%     for i = NeuralNetworkInput(1).numLayers:-1:1
%         if(i == NeuralNetworkInput(1).numLayers)
%             NeuralNetworkInput(1).layerData(i).delta = 2*(NeuralNetworkInput(1).layerData(i).X - outputVector)*hw12_q1_bwdTransformFunction(NeuralNetworkInput(1).layerData(i).X, transformFunction);
%         else
%             %thetaDash = ones(size(NeuralNetwork(1).layerData(i).X)) - (NeuralNetwork(1).layerData(i).X).*(NeuralNetwork(1).layerData(i).X);
%             %thetaDash = hw12_q1_bwdTransformFunction(NeuralNetworkInput(1).layerData(i).X, transformFunction);
%             thetaDash = ones(size(NeuralNetworkInput(1).layerData(i).X));
%             prevDelta = repmat(NeuralNetworkInput(1).layerData(i + 1).delta', NeuralNetworkInput(1).layerData(i).numHiddenNodes, 1);
%             NeuralNetworkInput(1).layerData(i).delta = thetaDash.*(NeuralNetworkInput(1).layerData(i + 1).W(2:end,:).*prevDelta);
%         end
%     end

%Loop unrolling
    NeuralNetworkInput(1).layerData(2).delta = 2*(NeuralNetworkInput(1).layerData(2).X - outputVector)*(ones(size(NeuralNetworkInput(1).layerData(2).X)));
    thetaDash = ones(size(NeuralNetworkInput(1).layerData(1).X));
    prevDelta = repmat(NeuralNetworkInput(1).layerData(2).delta', NeuralNetworkInput(1).layerData(1).numHiddenNodes, 1);
    NeuralNetworkInput(1).layerData(1).delta = thetaDash.*(NeuralNetworkInput(1).layerData(2).W(2:end,:).*prevDelta);
    NeuralNetworkOutput = NeuralNetworkInput;
end