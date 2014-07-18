function [positions values] = NeuronExtractPoints(sim_result, pixelsize, stdev, threshold)
%extracts points from the simulated field and returns them in order of
%descending threshold value

%blur the volume
disp('Blurring...');
dev = [stdev stdev stdev]./pixelsize;
gauss_size = ceil(dev*4)
gauss_filter = ndgauss(gauss_size, dev);
blurred = imfilter(sim_result, gauss_filter);
SaveVOL(mat2gray(blurred)*255, '..\data\test_blurred.vol')
clear volume;
disp('done.');
%blurred = sim_result;
%find local maxima
disp('Finding maxima...');

%[values maxpos] = MinimaMaxima3D(blurred);
[values maxpos] = MinimaMaxima3D(sim_result);
disp('done.');

%create and save the volume
max_volume = uint8(zeros(size(blurred)));

for i = 1:size(values)
    max_volume(maxpos(i, 1), maxpos(i, 2), maxpos(i, 3)) = 255;
end
SaveVOL(max_volume, '..\data\test_maxima.vol');

vol_size = size(sim_result);
positions(:, 1) = maxpos(:, 1)/vol_size(1);
positions(:, 2) = maxpos(:, 2)/vol_size(2);
positions(:, 3) = maxpos(:, 3)/vol_size(3);

%eliminate values below threshold
for i = 1:size(values, 1)
    if values(i) >= threshold
        index = i;
    end
end

values = values(1:index);
positions = positions(1:index, :);