function [coeff meanvector latent] = NeuronPCA(filenames, percent_zeros, resample, pixelsize)

global template_size;

stdev = 1;

ballpark_samples = 150000;
desired_zeros = ballpark_samples*percent_zeros;
desired_nonzeros = ballpark_samples*(1.0 - percent_zeros);

num_PCA_files = size(filenames, 2);
final_source_array = [];
for f=1:num_PCA_files
    vol_filename = [filenames{f} '.vol'];
    point_filename = [filenames{f} '.cel'];
    disp('Loading tissue data...');
    %load the tissue volume
    volume = LoadVOL(vol_filename);

    if min(resample) > 0
        kernel = ndgauss(resample*2, resample./2.0);
        volume = imfilter(volume, kernel);
        volume = volume(1:resample(1):size(volume, 1), 1:resample(2):size(volume, 2), 1:resample(3):size(volume, 3));
    end

    %convert cells to integers
    point_list = OBJToPoints(point_filename);
    point_list(:, 1) = round(point_list(:, 1)*size(volume, 1));
    point_list(:, 2) = round(point_list(:, 2)*size(volume, 2));
    point_list(:, 3) = round(point_list(:, 3)*size(volume, 3));
    disp('done.');

    %create the matrix of target values
    %This is a volume with the ANN target value at each voxel
    %and is constructed by placing a normalized gaussian at each
    %cell location.

    target_volume = CreateTargetVolume(pixelsize, stdev, point_list, size(volume)) == 1;

    SaveVOL(mat2gray(target_volume)*255, '..\data\test_PCA_target.vol');

    disp('Constructing mask...');
    %initialize the mask with the target volume
    mask_volume = zeros(size(target_volume));
    %find the probability of using a zero value
    sx = size(volume, 1);
    sy = size(volume, 2);
    sz = size(volume, 3);
    pmin = floor([template_size(1)/2.0+1 template_size(2)/2.0+1 template_size(3)/2.0+1]);
    pmax = floor([sx - template_size(1)/2.0, sy - template_size(2)/2.0, sz - template_size(3)/2.0]);

    num_nonzeros = nnz(target_volume(pmin(1):pmax(1), pmin(2):pmax(2), pmin(3):pmax(3)));
    num_zeros = (pmax(1) - pmin(1))*(pmax(2) - pmin(2))*(pmax(3) - pmin(3)) - num_nonzeros;

    probability_zero = desired_zeros/num_zeros;
    probability_nonzero = desired_nonzeros/num_nonzeros;

    for z = pmin(3):pmax(3)
        for y = pmin(2):pmax(2)
            for x = pmin(1):pmax(1)
                if target_volume(x, y, z) > 0 && rand() < probability_nonzero
                    mask_volume(x, y, z) = 1;
                end
                if target_volume(x, y, z) == 0 && rand() < probability_zero
                    mask_volume(x, y, z) = 1;
                end
            end
        end
    end
    disp('done.');

    %create the sample array based on the input and target volumes
    disp('Allocating sample array...');

    %allocate the source and target arrays
    mask_nnz = nnz(mask_volume);
    target_array = zeros(1, mask_nnz);
    disp('done');

    disp('Filling sample array...');
    source_array = GetMaskedVectors(volume, mask_volume);
    i = 1;
    for z = pmin(3):pmax(3)
        for y = pmin(2):pmax(2)
            for x = pmin(1):pmax(1)
                if mask_volume(x, y, z) > 0
                    target_array(i) = target_volume(x, y, z);
                    i = i+1;                        
                end
            end
        end
    end
    disp('done.');
    final_source_array = [final_source_array source_array];
    size(final_source_array)
end


clear volume;
clear target_volume;
clear mask_volume;
disp('Calculating principle components...');
%calculate principle components and scores
[coeff scores latent] = princomp(final_source_array.');
disp('Number of values used: ');
disp(size(scores, 1));
meanvector = mean(source_array.');
clear source_array;
disp('done.');