function net = buildNetconv4(options)
%=========================================================================%
% Build the neural network.
%=========================================================================%

switch options.type
    case 'MLP1'
        input = imageInputLayer(options.inputSize, 'Name', 'input');
        fc1 = convolution2dLayer([16 3],16, 'Padding','same', 'Name','fc1');
        BN1 = batchNormalizationLayer('Name','BN1');
        relu1 = reluLayer('Name','relu1');
        drop1 = dropoutLayer(0.4, 'Name', 'drop1');
        fc2 = convolution2dLayer([8 3],32, 'Padding','same', 'Name','fc2');
        BN2 = batchNormalizationLayer('Name','BN2');
        relu2 = reluLayer('Name','relu2');
        drop2 = dropoutLayer(0.4, 'Name', 'drop2');
        fc3 = convolution2dLayer([4 2],64, 'Padding','same','Name', 'fc3');
        BN3 = batchNormalizationLayer('Name','BN3');
        relu3 = reluLayer('Name','relu3');
        drop3 = dropoutLayer(0.4, 'Name', 'drop3');
        fc4 = convolution2dLayer([3 2],128,'Padding', 'same', 'Name','fc4');
        BN4 = batchNormalizationLayer('Name','BN4');
        relu4 = reluLayer('Name','relu4');
        drop4 = dropoutLayer(0.4, 'Name', 'drop4');
        fc5 = convolution2dLayer([2 1],256,'Padding','same', 'Name','fc5');
        BN5 = batchNormalizationLayer('Name','BN5');
        relu5 = reluLayer('Name','relu5');
        drop5 = dropoutLayer(0.4, 'Name', 'drop5');
        fc6 = convolution2dLayer([1 1],256,'Padding','same', 'Name','fc6');
        BN6 = batchNormalizationLayer('Name','BN6');
        relu6 = reluLayer('Name','relu6');
        drop6 = dropoutLayer(0.4, 'Name', 'drop6');
        fc7 = fullyConnectedLayer(options.numAnt(2), 'Name', 'fc7');
        sfm = softmaxLayer('Name','sfm');
        classifier = classificationLayer('Name','classifier');

        layers = [
                  input
                  fc1
                  BN1
                  relu1
                  drop1
                  fc2
                  BN2
                  relu2
                  drop2
                  fc3
                  BN3
                  relu3
                  drop3
                  %{fc4
                  %BN4
                  %relu4
                 %drop4
                  %fc5
                  %BN5
                 %relu5
                  %drop5
                  %fc6
                  %BN6
                 %relu6
                  %drop6
                  fc7
                  sfm
                  classifier
                 ];
        net = layerGraph(layers);

end
