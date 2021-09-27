function [dataset,options] = dataPrep_tarin(fileName,options)
%=========================================================================%
% dataPrep is data preparing function.
% INPUTS:
%	fileName: a single-field data struct with data
%		  file paths.
%	options: a MATLAB structure with the experiment settings
% OUTPUTS:
%	dataset: dataset structure for training and validataion data
%	options: updated options structure
%=========================================================================%

len = length(fileName);
d = {};
loc = {};
labels = {};
for i = 1:len% Normalize data
	rawData = load(fileName(i).name)
	x = rawData.channel;
    %loc(i) = {rawData.userLoc};
    if strcmp( options.case, 'NLOS' ) 
        labels(i) = {rawData.labels};
    end
	d(i) = {x};
end

dataset.data = d;
dataset.labels = labels;
clear d loc labels

% Shuffling data:
% ---------------
options.numSamples = size( dataset.data{1},3 );
shuffledInd = randperm(options.numSamples);
options.shuffledInd = shuffledInd;
for i = 1:len
    dataset.data{i} = dataset.data{i}(:,:,shuffledInd);
end

% Divide data:
% ------------
numTrain = floor( (1 - options.valPer)*options.numSamples );
options.numOfTrain = numTrain;
options.numOfVal = options.numSamples - options.numOfTrain;
sub6Train = dataset.data{1}(:,:,1:numTrain);% Sub-6 training channels
sub6Val = dataset.data{1}(:,:,numTrain+1:end);% Sub-6 validation channels
if len > 1
    highTrain = dataset.data{2}(:,:,1:numTrain);% High training channels
    highVal = dataset.data{2}(:,:,numTrain+1:end);% High validation channels
end

% Compute data statistics:
% ------------------------
abs_value = abs( sub6Train );
max_value(1) = max(abs_value(:));
if len > 1
    abs_value = abs( highTrain );
    max_value(2) = max(abs_value(:));
end
options.dataStats = max_value;

%------------------------------------------------------
% Prepare inputs:
% ---------------
sub6Train = sub6Train/options.dataStats(1);% normalize training data
sub6Val = sub6Val/options.dataStats(1);% normalize validation data
X = zeros(1,options.numAnt(2),options.numSub1,options.numOfTrain);
Y = zeros(1,options.numAnt(2),options.numSub1,length(sub6Val));
if options.noisyInput
    % Noise power
    NF=5;% Noise figure at the base station
    Pr=30;
    BW=options.bandWidth*1e9; % System bandwidth in Hz
    noise_power_dB=-204+10*log10(BW/options.numSub)+NF; % Noise power in dB
    noise_power=10^(.1*(noise_power_dB));% Noise power
    Pn_r=(noise_power/options.dataStats(1)^2)/2;
    Pn=Pn_r/(10^(.1*(options.transPower-Pr)));
    SNR = 25*rand(1)-14;
    % Adding noise
    fprintf(['Corrupting channel measurements with ' num2str(Pn) '-variance Gaussian\n'])
    noise_samples = sqrt(Pn)*randn(size(sub6Train));% Zero-mean unity-variance noise
    for j = 1:size(sub6Train,3)
        
        h = awgn(sub6Train(:,:,j),SNR,'measured');
        h1 = abs(CSI2ADP_theta_N(h,options.numAnt(2),options.numSub1,options.numAnt(1),options.numSub)/sqrt(options.numAnt(1)*options.numSub));
        sub6Train1(:,:,j) = h1;
        
    end
    noise_samples = sqrt(Pn)*randn(size(sub6Val));
    for j = 1:size(sub6Val,3)
        h = awgn(sub6Val(:,:,j),SNR,'measured');
        h1 = abs(CSI2ADP_theta_N(h,options.numAnt(2),options.numSub1,options.numAnt(1),options.numSub)/sqrt(options.numAnt(1)*options.numSub));
        sub6Val1(:,:,j) = h1;
    end 
else
    fprintf('Clean channel measurements')
end
for i = 1:options.numOfTrain
    x = sub6Train1(:,:,i);
    X(1,:,:,i) = x;
end

for i = 1:length(sub6Val)
    y = sub6Val1(:,:,i);
    Y(1,:,:,i) = y;
end
dataset.inpTrain = X;
dataset.inpVal = Y;

%-----------------------------------------------------
% Prepare outputs:
% ----------------


highTrain = highTrain(1:options.numAnt(2),1:options.numSub,:)/options.dataStats(2);
highVal = highVal(1:options.numAnt(2),1:options.numSub,:)/options.dataStats(2);
dataset.highFreqChTrain = highTrain;% 
dataset.highFreqChVal = highVal;% 
W = options.codebook;
value_set = 1:size(W,2);
for i = 1:options.numOfTrain
    H = highTrain(:,:,i);
    rec_power = abs( H'*W ).^2;
    rate_per_sub = log2( 1 + rec_power );
    rate_ave = sum(rate_per_sub,1)/options.numSub;
    [r,ind] = max( rate_ave, [], 2 );
    beam_ind(i,1) = ind;
    max_rate(i,1) = r;
end
dataset.labelTrain = categorical( beam_ind, value_set );
dataset.maxRateTrain = max_rate;
beam_ind = [];
max_rate = [];
for i = 1:options.numOfVal
    H = highVal(:,:,i);
    rec_power = abs( H'*W ).^2;
    rate_per_sub = log2( 1 + rec_power );
    rate_ave = sum(rate_per_sub,1)/options.numSub;
    [r,ind] = max( rate_ave, [], 2 );
    beam_ind(i,1) = ind;
    max_rate(i,1) = r;
end
dataset.labelVal = categorical( beam_ind, value_set );
dataset.maxRateVal = max_rate;
dataset = rmfield(dataset,'data');
dataset = rmfield(dataset,'labels');