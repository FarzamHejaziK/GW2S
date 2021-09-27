function [dataset,options] = dataPrep_test(fileName,options)
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

dataset.data = d;
dataset.userLoc = loc;
dataset.labels = labels;
clear d labels
labels = {};

for i = 3:4% Normalize data
	rawData = load(fileName(i).name)
	x = rawData.channel;
    if strcmp( options.case, 'NLOS' ) 
        labels(i-2) = {rawData.labels};
    end
	d(i-2) = {x};
end

testdataset.data = d;
testdataset.userLoc = loc;
testdataset.labels = labels;
clear d loc labels


% Shuffling data:
% ---------------

options.numOfVal = length(testdataset.labels);

sub6Val = testdataset.data{1};% Sub-6 validation channels

if len > 1
    highVal = testdataset.data{2};% High validation channels
end


abs_value = abs( sub6Val );
max1_value(1) = max(abs_value(:));
if len > 1
    abs_value = abs( highVal );
    max1_value(2) = max(abs_value(:));
end


options.dataStats1 = max1_value;

%------------------------------------------------------
% Prepare inputs:
% ---------------

sub6Val = sub6Val/options.dataStats1(1);% normalize validation data

Y = zeros(1,options.numAnt(2),options.numSub1,length(sub6Val));

if options.noisyInput
    SNR = options.SNR;
    for j = 1:size(sub6Val,3)
        h = awgn(sub6Val(:,:,j),SNR,'measured');
        h1 = abs(CSI2ADP_theta_N(h,options.numAnt(2),options.numSub1,options.numAnt(1),options.numSub)/sqrt(options.numAnt(1)*options.numSub));
        sub6Val1(:,:,j) = h1;
    end 
else
    fprintf('Clean channel measurements')
end

%}
for i = 1:length(sub6Val)
    y = sub6Val1(:,:,i);
    Y(1,:,:,i) = y;
end

dataset.inpVal = Y;

%-----------------------------------------------------
% Prepare outputs:
% ----------------


highVal = highVal(1:options.numAnt(2),1:options.numSub,:)/options.dataStats1(2);
dataset.highFreqChVal = highVal;% 
W = options.codebook;
value_set = 1:size(W,2);

beam_ind = [];
max_rate = [];
for i = 1:length(highVal)
    H = highVal(:,:,i);
    rec_power = abs( H'*W ).^2;
    rate_per_sub = log2( 1 +rec_power*(10^(options.SNR/10)));
    rate_ave = sum(rate_per_sub,1)/options.numSub;
    [r,ind] = max( rate_ave, [], 2 );
    beam_ind(i,1) = ind;
    max_rate(i,1) = r;
end
dataset.labelVal = categorical( beam_ind, value_set );
dataset.maxRateVal = max_rate;
dataset = rmfield(dataset,'data');
dataset = rmfield(dataset,'labels');
dataset = rmfield(dataset, 'userLoc');
