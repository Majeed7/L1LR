
% Add this file to the root of L1General
%%%%%%%%%%%%%%%%%%%%%% Set-up Problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
% Load Binary Classification Data
[y, data] = libsvmread('.\datasets\splice');
[y_t, data_t] = libsvmread('.\datasets\splice.t');
exp_name = 'splice';
%% preprocess
data = full(data);
% [r,c] = size(data);
% data = [data, zeros(r, 123-c)];

data_t = full(data_t);
% [r,c] = size(data_t);
% data_t = [data_t, zeros(r, 123-c)];

y_t(y_t == 0) = -1;
y(y == 0) = -1;
% Scale feature to be N(0,1)
X = standardizeCols(data(:,1:end-1));
[nInstances,nVariables] = size(X);

x_t = standardizeCols(data_t(:,1:end-1));
[nInstances_t,~] = size(x_t);
% Add bias
X = [ones(nInstances,1) X];
nVariables = nVariables + 1;

x_t = [ones(nInstances_t,1) x_t];
% Use Logistic Regression loss function
loss = @LogisticLoss;

% Arguments for Logistic Regression are X and y
lossArgs = {X,y};

% Set all variables to be initially 0
w_init = zeros(nVariables,1);

% Set lambda (higher values yield more regularization/sparsity)
lambda = 1;

% Penalize all variables except bias
lambdaVect = [0;lambda*ones(nVariables-1,1)];

% Use default options
% (some options get changed to invoke individual methods)
global_options = [];

% Choose whether to require Hessian information explicitly
global_options.order = 2; % Change this to 1 to use BFGS versions

% To get more accurate timing, you can turn off verbosity:
global_options.verbose = 0;

global_options.maxIter = 100;
global_options.optTol = 1e-6;
global_options.threshold = 1e-7;
% To run full script without stop:
without_stop = 1;
acc = [];
exe_time = [];
method_names = {};
y_scores = {};
%% %%%%%%%%%%%%%%%%%%%% Run Methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nNext Algorithm: Gauss-Seidel...\n');
pause(0.5);
optimFunc = @L1GeneralCoordinateDescent;
options = global_options;
tic
wGS = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wGS, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'Gauss-Seidel';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: Shooting...\n');
pause(0.5);
optimFunc = @L1GeneralCoordinateDescent;
options = global_options;
options.mode = 1; % Turns on Shooting instead of Gauss-Seidel
tic
wShoot = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wShoot, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'Shooting';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: Gauss-Southwell...\n');
pause(0.5);
optimFunc = @L1GeneralCoordinateDescent;
options = global_options;
options.mode = 2; % Use Gauss-Southwell
tic
wShoot = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wShoot, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'Gauss-Southwell';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: Grafting...\n');
pause(0.5);
optimFunc = @L1GeneralGrafting;
options = global_options;
options.mode = 1;
tic
wGraft = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wGraft, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'Grafting';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: SubGradient...\n');
pause(0.5);
optimFunc = @L1GeneralSubGradient;
options = global_options;
tic
wSubGrad = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wSubGrad, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
fprintf('\n(paused, press any key to start next algorithm)\n');
method_names{end+1} = 'SubGradient';
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: Max-K SubGradient...\n');
pause(0.5);
optimFunc = @L1GeneralSubGradient;
options = global_options;
options.k = 1;
tic
wMaxK = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wMaxK, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'Max-K SubGradient';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: epsL1...\n');
pause(0.5);
optimFunc = @L1GeneralUnconstrainedApx;
options = global_options;
options.mode = 1; % Turns on epsL1 instead of smoothL1
options.cont = 0; % Turn off continuation
tic
wEpsL1 = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wEpsL1, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'epsL1';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
if global_options.order == 2
	fprintf('\nNext Algorithm: Log-Barrier...\n');
	pause(0.5);
	optimFunc = @L1GeneralUnconstrainedApx;
	options = global_options;
	options.mode = 3; % Turns on using non-negative Log-Barrier instead of SmoothL1
	tic
	wLogBarrier = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
	exe_time(end+1) = toc;
	[acc(end+1),y_scores{end+1}] = predict(wLogBarrier, x_t, y_t);
	fprintf('Accuracy = %f\n', acc(end));
    method_names{end+1} = 'Log-Barrier';
	fprintf('\n(paused, press any key to start next algorithm)\n');
	if without_stop == 0
		pause;
	end
end
%%
fprintf('\nNext Algorithm: SmoothL1 (short-cut)...\n');
pause(0.5);
optimFunc = @L1GeneralUnconstrainedApx;
options = global_options;
options.cont = 0; % Turn off continuation
tic
wSmoothL1sc = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wSmoothL1sc, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'SmoothL1 (short-cut)';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: SmoothL1 (continuation)...\n');
pause(0.5);
optimFunc = @L1GeneralUnconstrainedApx;
options = global_options;
tic
wSmoothL1ct = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wSmoothL1ct, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'SmoothL1 (continuation)';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: EM...\n');
pause(0.5);
optimFunc = @L1GeneralIteratedRidge;
options = global_options;
tic
wEM = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wEM, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'EM';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
if exist('quadprog') == 2
    fprintf('\nNext Algorithm: SQP...\n');
    pause(0.5);
    optimFunc = @L1GeneralSequentialQuadraticProgramming;
    options = global_options;
    tic
    wSQP = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
    exe_time(end+1) = toc;
    [acc(end+1),y_scores{end+1}] = predict(wSQP, x_t, y_t);
    fprintf('Accuracy = %f\n', acc(end));
    method_names{end+1} = 'SQP';
    fprintf('\n(paused, press any key to start next algorithm)\n');
    if without_stop == 0
		pause;
	end
else
    fprintf('\nquadprog not found, skipping SQP\n');
end
%%
fprintf('\nNext Algorithm: ProjectionL1...\n');
pause(0.5);
optimFunc = @L1GeneralProjection;
options = global_options;
tic
wProj = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wProj, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'ProjectionL1';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: InteriorPoint...\n');
pause(0.5);
optimFunc = @L1GeneralPrimalDualLogBarrier;
options = global_options;
tic
wIP = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wIP, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'InteriorPoint';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end
%%
fprintf('\nNext Algorithm: Orthant-Wise...\n');
pause(0.5);
optimFunc = @L1GeneralOrthantWise;
options = global_options;
tic
wOrthant = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wOrthant, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'Orthant-Wise';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end

if global_options.order == 2
	fprintf('\nNext Algorithm: Pattern-Search...\n');
	pause(0.5);
	optimFunc = @L1GeneralPatternSearch;
	options = global_options;
	tic
	wPS = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
	exe_time(end+1) = toc;
	[acc(end+1),y_scores{end+1}] = predict(wPS, x_t, y_t);
	fprintf('Accuracy = %f\n', acc(end));
    method_names{end+1} = 'Pattern-Search';
	fprintf('\n(paused, press any key to start next algorithm)\n');
	if without_stop == 0
		pause;
	end
end

fprintf('\nNext Algorithm: Projected SubGradient...\n');
pause(0.5);
optimFunc = @L1GeneralProjectedSubGradient;
options = global_options;
tic
wPSG = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
exe_time(end+1) = toc;
[acc(end+1),y_scores{end+1}] = predict(wPSG, x_t, y_t);
fprintf('Accuracy = %f\n', acc(end));
method_names{end+1} = 'Projected SubGradient';
fprintf('\n(paused, press any key to start next algorithm)\n');
if without_stop == 0
	pause;
end

if global_options.order == 1
	fprintf('\nNext Algorithm: Projected SubGradient BB...\n');
	pause(0.5);
	optimFunc = @L1GeneralProjectedSubGradientBB;
	options = global_options;
	tic
	wPSG = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
	exe_time(end+1) = toc;
	[acc(end+1),y_scores{end+1}] = predict(wPSG, x_t, y_t);
	fprintf('Accuracy = %f\n', acc(end));
    method_names{end+1} = 'Projected SubGradient BB';
	fprintf('\n(paused, press any key to start next algorithm)\n');
	if without_stop == 0
		pause;
	end
	fprintf('\nNext Algorithm: Accelerated Soft-Thresholding...\n');
	pause(0.5);
	optimFunc = @L1GeneralCompositeGradientAccelerated;
	options = global_options;
	tic
	wPSG = optimFunc(loss,w_init,lambdaVect,options,lossArgs{:});
	exe_time(end+1) = toc;
	[acc(end+1),y_scores{end+1}] = predict(wPSG, x_t, y_t);
	fprintf('Accuracy = %f\n', acc(end));
    method_names{end+1} = 'Accelerated Soft-Thresholding';
	fprintf('\n(paused, press any key to start next algorithm)\n');
	if without_stop == 0
		pause;
	end
end
results = {};
results{1} = acc';
results{2} = exe_time';
results{3} = method_names;
results{4} = y_scores;

save(['result_' exp_name '.mat'], 'results');