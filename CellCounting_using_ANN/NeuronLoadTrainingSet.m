function [inputs targets] = NeuronLoadTrainingSet(vol_filename, point_filename, percent_zeros, PC, meanvector, resample, pixelsize)

global template_size;

pixelsize = pixelsize .* resample;
stdev = 5;
%vol_filename = '..\data\training01.vol';
%point_filename = '..\data\training01.cel';

ballpark_samples = 50000;
desired_zeros = ballpark_samples*percent_zeros;
desired_nonzeros = ballpark_samples*(1.0 - percent_zeros);

disp('Loading tissue data...');
%load the tissue volume
volume = double(LoadVOL(vol_filename));

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

target_volume = CreateTargetVolume(pixelsize, stdev, point_list, size(volume));
SaveVOL(target_volume*255, '..\data\test_target_volume.vol');

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
num_zero_vectors = 0;
num_nonzero_vectors = 0;

for z = pmin(3):pmax(3)
    for y = pmin(2):pmax(2)
        for x = pmin(1):pmax(1)
% csung - increase vectors...
%             if target_volume(x, y, z) > 0 && rand() < probability_nonzero
%                 mask_volume(x, y, z) = 1;
%                 num_nonzero_vectors = num_nonzero_vectors +1;
%             end
%             if target_volume(x, y, z) == 0 && rand() < probability_zero
%                 mask_volume(x, y, z) = 1;
%                 num_zero_vectors = num_zero_vectors + 1;
%             end
%%
            if (target_volume(x, y, z) > 0) && (rand()/2 < probability_nonzero)
                mask_volume(x, y, z) = 1;
                num_nonzero_vectors = num_nonzero_vectors +1;
            elseif (target_volume(x, y, z) == 0) && (rand()/2 < probability_zero)
                mask_volume(x, y, z) = 1;
                num_zero_vectors = num_zero_vectors + 1;
            end
        end
    end
end
% csung - add center points into the inputs.
for c_i=1:size(point_list, 1);
    p = point_list(c_i, :);
    if(p(1)>pmin(1)&&p(1)<pmax(1))&&(p(2)>pmin(2)&&p(2)<pmax(2))&&(p(3)>pmin(3)&&p(3)<pmax(3))
        mask_volume(p(1),p(2),p(3)) = 1;
        num_nonzero_vectors = num_nonzero_vectors +1;
    end
end
num_zero_vectors
num_nonzero_vectors
disp('done.');


%create the sample array based on the input and target volumes
disp('Allocating sample array...');

%allocate the source and target arrays
mask_nnz = nnz(mask_volume);
inputs = GetMaskedVectors(volume, mask_volume, PC, meanvector);
targets = zeros(1, mask_nnz);
disp('done');

disp('Filling sample array...');
i = 1;
for z = pmin(3):pmax(3)
    for y = pmin(2):pmax(2)
        for x = pmin(1):pmax(1)
            if mask_volume(x, y, z) > 0

                %p0 = floor([x y z] - floor(template_size./2.0));
                %p1 = floor(p0 + template_size - 1);

                %subvolume = volume(p0(1):p1(1), p0(2):p1(2), p0(3):p1(3));
                %[slice_x, slice_y, slice_z] = OrthoSlices(subvolume);
                %inputs(:, i) = ([slice_x(:); slice_y(:); slice_z(:)].' - meanvector) * PC;

                targets(i) = target_volume(x, y, z);
                i = i+1;                        
            end
        end
    end
end
disp('done.');


