%================================================================================%
% Main script for training and testing a DL model to predict mmWave (28   
% GHz) beam indecies from sub-6 GHz channels. The script assumes the data
% provided in the form of four (.mat) files: 
%   - dataFile1: Sub-6 train data
%   - dataFile2: mmWave train data   
%   - dataFile2: Sub-6 train data
%   - dataFile1: mmWave test data
% -------------------------------------------------------------------------------
% This File is an updated version of https://github.com/malrabeiah/Sub6-Preds-mmWave/blob/master/main.m
%=================================================================================%
clc
clear
close all

tx_power = [-37.3375  -32.3375  -27.3375  -22.3375  -17.3375  -12.3375   -7.3375];
snr_db = [-10.0586   -5.0586   -0.0586    4.9414    9.9414   14.9414   19.9414];
num_ant = [4];% number of sub-6 antennas
num_ant_mm = [64];% number of mmWave antennas
accuracy_top1 = zeros(length(num_ant_mm), length(tx_power));
accuracy_top3 = zeros(length(num_ant_mm), length(tx_power));
ave_rate_top1 = zeros(length(num_ant_mm), length(tx_power));
ave_rate_top3 = zeros(length(num_ant_mm), length(tx_power));
ave_upper = zeros(length(num_ant_mm), length(tx_power));
options.figCount = 0;
options.type = 'MLP1';
options.case = 'LOS';
options.expTag = ['Results\' options.type '_NDforTest' '_test1_variableSNR' '10number2'];
options.top_n = 3;
options.valAccuracy = zeros(length(num_ant_mm),length(tx_power));
options.normMethod = 'perDataset';
options.gpuInd = 1;
fprintf('Experiment: %s\n', options.expTag);
ant = 1;
%for ant = 1:length(num_ant_mm)% number of antennas to loop over
	fprintf('Number of sub-6 antennas: %d and number of mmWave antennas: %d\n', num_ant(1), num_ant_mm(ant))
    [W,~] = UPA_codebook_generator(1,num_ant_mm(ant),1,1,1,1,0.5);% Beam codebook
	options.codebook = W;
	options.numAnt = [num_ant(1), num_ant_mm(ant)];
	options.numSub = 64;
    options.numSub1 = 64;
	options.inputDim = [options.numAnt(1) options.numSub];
	options.valPer = 0.01;
	options.inputSize = [1,options.numAnt(2),options.numSub1];
	options.noisyInput = true;
	options.bandWidth = 0.5;
	options.dataFile1 ='Data/train/sub6Train_org_4_64.mat';% The path to the sub-6 data file
    options.dataFile2 ='Data/train/mmTrain_org_64_64.mat';% The path to the mmWave data file
    options.testFile1 ='Data/test/sub6test_LOSB_4_64.mat';
    options.testFile2 ='Data/test/mmtest_LOSB_4_64.mat';
    if isempty(options.dataFile1)
		error('Please provide a sub-6 data file!');
	elseif isempty(options.dataFile2)
		error('Please provide a mmWave data file');
	end
	% tarining settings
	options.solver = 'adam';
	options.learningRate = 3e-4;
	options.schedule = 40;
	options.dropLR = 'piecewise';
	options.dropFactor = 0.3;
	options.maxEpoch = 100;
	options.batchSize =1000;
	options.verbose = 1;
	options.verboseFrequency = 50;
    options.valFreq = 100;
	options.shuffle = 'every-epoch';
	options.weightDecay = 1e-6;
	options.progPlot = 'none';
    
    options.transPower = 1;
    
    options.SNR = 10.2;
    
		fileName = struct('name',{options.dataFile1,options.dataFile2});
        [dataset,options] = dataPrep_train(fileName, options);
    net = buildNetconv4(options);    
		trainingOpt = trainingOptions(options.solver,...
		    'InitialLearnRate',options.learningRate,...
		    'LearnRateSchedule',options.dropLR, ...
		    'LearnRateDropFactor',options.dropFactor, ...
		    'LearnRateDropPeriod',options.schedule, ...
		    'MaxEpochs',options.maxEpoch, ...
		    'L2Regularization',options.weightDecay,...
		    'Shuffle', options.shuffle,...
		    'MiniBatchSize',options.batchSize, ...
		    'ValidationData', {dataset(1).inpVal, dataset(1).labelVal},...
            'ValidationFrequency', options.valFreq,...
		    'Verbose', options.verbose,...
		    'verboseFrequency', options.verboseFrequency,...
		    'Plots',options.progPlot);
        
       [trainedNet, trainInfo] = trainNetwork(dataset.inpTrain, dataset.labelTrain,net,trainingOpt);
	%{	
    net =  load('C:\Users\farza\OneDrive - University of Central Florida\Convforgen_4_64 (1)\Nets\net1.mat');  
    trainedNet = load('C:\Users\farza\OneDrive - University of Central Florida\Convforgen_4_64 (1)\Nets\trainedNet1.mat');
    trainInfo = load('C:\Users\farza\OneDrive - University of Central Florida\Convforgen_4_64 (1)\Nets\trainInfo1.mat');
	%}
    for p = 1:length(tx_power)

        fprintf('Pt = %4.2f (dBm)\n', tx_power(p))

		% Prepare dataset:
		% ----------------
        options.SNR = snr_db(p);
		options.transPower = tx_power(p);
        fileName = struct('name',{options.dataFile1,options.dataFile2,options.testFile1,options.testFile2});
		[dataset,options] = dataPrep_test(fileName, options);

		% Test network:
		% -------------

		X = dataset.inpVal;
        Y = single( dataset.labelVal );
        options.numOfVal = size(X,4);
        [pred,score] = trainedNet.classify(X);
        pred = single( pred );
        highFreqCh = dataset.highFreqChVal;
        hit = 0;
        hit1 = 0;
        hit2 = 0;
        options.numOfVal = size(X,4);
        for user = 1:size(X,4)
            % Top-1 average rate
            H = highFreqCh(:,:,user);    
            w = W(:,pred(user));
            rec_power = abs( H'*w ).^2;
            rate_per_sub = log2( 1 +rec_power*(10^(options.SNR/10)));
            rate_top1(user) = sum(rate_per_sub)/options.numSub;

            % Top-3 accuracy
            rec_power = abs( H'*W ).^2;
            rate_per_sub = log2( 1 +rec_power*(10^(options.SNR/10)));
            ave_rate_per_beam = mean( rate_per_sub, 1);
            [~,ind] = max(ave_rate_per_beam);% the best beam
            [~,sort_ind] = sort( score(user,:), 'descend' );
            three_best_beams = sort_ind(1:options.top_n);
            first_best_beams = sort_ind(1:1);
            five_best_beams = sort_ind(1:5);
            if any( three_best_beams == ind )
                hit = hit + 1;
            end
            if any( first_best_beams == ind )
                hit1 = hit1 + 1;
            end
            
            if any( five_best_beams == ind )
                hit2 = hit2 + 1;
            end
            
            % Top-3 average rate
            rec_power = abs( H'*W(:,three_best_beams) ).^2;
            rate_per_sub = log2( 1 +rec_power*(10^(options.SNR/10)));
            ave_rate_per_beam = mean(rate_per_sub,1);
            rate_top3(user) = max( ave_rate_per_beam );
            rec_power1 = abs( H'*W(:,five_best_beams) ).^2;
            rate_per_sub1 = log2( 1 + rec_power1*(10^(options.SNR/10)));
            ave_rate_per_beam1 = mean(rate_per_sub1,1);
            rate_top5(user) = max( ave_rate_per_beam1);

        end
        accuracy_top1(ant,p) = 100*(hit1/options.numOfVal);
        accuracy_top3(ant,p) = 100*(hit/options.numOfVal);
        accuracy_top5(ant,p) = 100*(hit2/options.numOfVal);
        ave_rate_top1(ant,p) = mean(rate_top1);
        ave_rate_top3(ant,p) = mean(rate_top3);
        ave_rate_top5(ant,p) = mean(rate_top5);
        ave_upper(ant,p) = mean(dataset.maxRateVal);
        fprintf('Top-1 and Top-3 and Top-5 rates: %5.3f & %5.3f & %5.3f. Upper bound: %5.3f\n', ave_rate_top1(ant,p),ave_rate_top3(ant,p),ave_rate_top5(ant,p),...
                     mean( dataset.maxRateVal ) );
        fprintf('Top-1 and Top-3 and Top-5 Accuracies: %5.3f%% & %5.3f%% & %5.3f%%\n', accuracy_top1(ant,p),accuracy_top3(ant,p), accuracy_top5(ant,p));

	end


% Save performance variables
variable_name = [options.expTag '_results'];
save(variable_name,'accuracy_top1','accuracy_top3','accuracy_top5','ave_rate_top1','ave_rate_top3','ave_rate_top5','ave_upper')
options.figCount = options.figCount+1;
fig1 = figure(options.figCount);
plot(snr_db, ave_rate_top1(1,:), '-b',...
     snr_db, ave_rate_top3(1,:), '-r',...
     snr_db, ave_rate_top5(1,:), '-c',...
     snr_db, ave_upper(1,:), '-.k');
xlabel('SNR (dB)');
ylabel('Spectral Efficiency (bits/sec/Hz)');
grid on
legend('Top-1 achievable rate','Top-3 achievable rate','Top-5 achievable rate','Upper bound')
name_file = [options.expTag 'fig'];
saveas(fig1,name_file)


name_file = [options.expTag 'trained_net'];
save(name_file,'trainedNet')

