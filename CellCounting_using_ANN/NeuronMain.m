%%
%set filenames
directory = '..\data\';
training_files = {};
training_files{size(training_files, 2)+1} = [directory 'training01'];
training_files{size(training_files, 2)+1} = [directory 'training03'];

%training_files{size(training_files, 2)+1} = [directory 'training04'];

%% useful
%training_files{size(training_files, 2)+1} = [directory 'training05'];
%training_files{size(training_files, 2)+1} = [directory 'training06'];

%training_files{size(training_files, 2)+1} = [directory 'training07'];
%training_files{size(training_files, 2)+1} = [directory 'training08a'];
%training_files{size(training_files, 2)+1} = [directory 'training08b'];


validation_files = {};
%validation_files{size(validation_files, 2)+1} = [directory 'training01'];
%validation_files{size(validation_files, 2)+1} = [directory 'training03'];
%validation_files{size(validation_files, 2)+1} = [directory 'training04'];
%validation_files{size(validation_files, 2)+1} = [directory 'training05'];
%validation_files{size(validation_files, 2)+1} = [directory 'training06'];
%validation_files{size(validation_files, 2)+1} = [directory 'training07'];
%validation_files{size(validation_files, 2)+1} = [directory 'training08a'];
%validation_files{size(validation_files, 2)+1} = [directory 'training08b'];
%%
%validation_files{size(validation_files, 2)+1} = [directory 'training08d'];
%validation_files{size(validation_files, 2)+1} = [directory 'training09'];
%validation_files{size(validation_files, 2)+1} = [directory 'training10a'];
validation_files{size(validation_files, 2)+1} = [directory 'training10b'];
validation_files{size(validation_files, 2)+1} = [directory 'training10c'];
%validation_files{size(validation_files, 2)+1} = [directory 'training10d'];
global template_size;
template_size = [15 13 10];
%template_size = [30 27 20];
%template_size = [7 6 5];

%global vector_length;
%vector_length = template_size(2)*template_size(3) + template_size(1)*template_size(3) + template_size(1)*template_size(2);
pixelsize = [1.2 1.4 2.0];
resample = [2 2 2];
num_components = 30;
percent_zero = 0.5;

radius = 5;


%ortho_size = [30 27 20];
%pixelsize = [0.6 0.7 1.0];
%resample = [1 1 1];

%%
%perform PCA
percent_zero = 0.0;
%calculate the ordered eigenvectors and eigenvalues.
[coeff meanvector latent] = NeuronPCA(training_files, percent_zero, resample, pixelsize);
%[PCA_x PCA_y PCA_z] = GetPCAImages(coeff(:, 1:num_components), template_size(1), template_size(2), template_size(3));


%load training set
percent_zero = 0.5;
inputs = [];
targets = [];

% only retain the top 'num_components' eigenvectors (i.e. the principal
% components) => coeff(:, 1:num_components)
num_training_files = size(training_files, 2);
for f=1:num_training_files
    volume_file = [training_files{f} '.vol'];
    cell_file = [training_files{f} '.cel'];
    [i0 t0] = NeuronLoadTrainingSet(volume_file, cell_file, percent_zero, coeff(:, 1:num_components), meanvector, resample, pixelsize);
    inputs = [inputs i0];
    targets = [targets t0];
end

size(inputs)
size(targets)

disp('Training...');

net = newff(inputs, targets, 5);

%methods for pre processing inputs and post processing outputs
%can be access in a 2-layer network by typing:
%The network will receive your inputs and normalize them using mapminmax.
%Then the data will be processed by the network and then
%the outputs will be transformed back to your original units.
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};

net = train(net, inputs, targets);
disp('done.');
%%
%simulation
all_states = [];
all_values = [];
all_undetected = 0;
threshold = 0;
num_validation_files = size(validation_files, 2);
for i=1:num_validation_files
    disp('Simulating...');
    volume_file = [validation_files{i} '.vol'];
    [result_volume] = NeuronSimulate(volume_file, net, coeff(:, 1:num_components), meanvector, resample);
    SaveVOL(mat2gray(result_volume, [0 1.0])*255, '..\data\test_sim.vol');
    disp('done.');
    
    disp('Computing Points...');
    maxima_time = tic;
    
    %get the points from the results image
    [cells, values] = NeuronExtractPoints(result_volume, pixelsize, 1, threshold);

    %save the ANN cel file
    PointsToOBJ(cells, '..\data\test_ANN.cel');
    %csung
    PointsToOBJ(cells, [validation_files{i} '.ann']);
    disp('done.');
    disp('Maxima time:');
    disp(toc(maxima_time));
    
    disp('Validating...');
    obj_filename = [validation_files{i} '.cel'];
    GT = OBJToPoints(obj_filename);
    
    %GT = RemoveBoundaryCells(GT, size(result_volume), template_size);
    %[cells values] = RemoveBoundaryCells(cells, size(result_volume), template_size, values);
    
    um_volume_size = [200 200 100].*[0.6 0.7 1.0];
    [states undetected] = ComparePoints(GT, cells, radius, um_volume_size);
    
    norm_template_size = template_size./[200 200 100];
    undetected = RemoveBoundaryPoints(undetected, norm_template_size);
    
    all_states = [all_states; states];
    all_values = [all_values; values];
    all_undetected = all_undetected + size(undetected, 1);
    disp('done.');
    
    disp('Saving cell files...');
    PointsToOBJ(undetected, '..\data\test_undetected.cel');
    false_positives = zeros(size(states, 1)-nnz(states), 3);
    p = 1;
    for li=1:size(states, 1)
        if states(li) == 0
            false_positives(p, :) = cells(li, :);
            p = p + 1;
        end
    end
    PointsToOBJ(false_positives, '..\data\test_falsepositive.cel');
    disp('done.');
end

[accuracy_curve AUC points peak_performance]= ComputeAccuracy(all_states, all_values, all_undetected);
AUC
peak_performance
accuracy_curve(peak_performance(3), :)


%%
best_threshold = 0;
best_point = 0;
for t=1:size(points, 1)
    if(accuracy_curve(t, 2) >= 0.9)
        best_threshold = points(t, 1);
        best_point = t;
    end
end

best_sensitivity = accuracy_curve(best_point, 2);

%csung - some time, best_point exceeds cells' index.
%PointsToOBJ(cells(1:best_point, :), '..\data\test_ANN.cel');
if(best_point > size(cells, 1))
    PointsToOBJ(cells(:,:), '..\data\test_ANN.cel');
else
    PointsToOBJ(cells(1:best_point, :), '..\data\test_ANN.cel');
end
%<-csung

%%
%[ROC AUC accuracy TPR FPR threshold] = ComputeROC(all_states, all_values, all_undetected);
%TPR
%FPR
%threshold
%AUC
%accuracy

