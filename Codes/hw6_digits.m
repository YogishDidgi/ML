%function hw6_digits
clear all
close all

fileID = fopen('ZipDigits.train','r');
formatSpec = '%f';
sizeA = [257, Inf];

A = fscanf(fileID,formatSpec,sizeA);
A = A';
mask_1 = find(A(:,1) == 1.0);
mask_5 = find(A(:,1) == 5.0);
Data_1 = A(mask_1, 2:257);
Data_5 = A(mask_5, 2:257);

sample_1 = Data_1(1, :);
sample_1 = reshape(sample_1, [16 16])';
figure,
subplot(1, 2, 1),imshow(1-sample_1, []);

sample_5 = Data_5(1, :);
sample_5 = reshape(sample_5, [16 16])';
subplot(1, 2, 2),imshow(1-sample_5, []);

intensity_1 = sum(Data_1, 2)/256;
intensity_5 = sum(Data_5, 2)/256;

Data_1_reshape = reshape(Data_1,[],16,16);
[m n o] = size(Data_1_reshape);
difference_1 = zeros(m,1);
for i=1:m
    sample_1(1:16,:) = Data_1_reshape(i,1:16,:);
    sample_1 = sample_1';
    sample_1_rot = rot90(sample_1,2);
    diff_1 = abs(sample_1 - fliplr(sample_1));
    diff_2 = abs(sample_1 - flipud(sample_1));
    diff_3 = abs(sample_1 - sample_1_rot);
    difference_1(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

Data_5_reshape = reshape(Data_5,[],16,16);
[m n o] = size(Data_5_reshape);
difference_5 = zeros(m,1);
for i=1:m
    sample_5(1:16,:) = Data_5_reshape(i,1:16,:);
    sample_5 = sample_5';
    sample_5_rot = rot90(sample_5,2);
    diff_1 = abs(sample_5 - fliplr(sample_5));
    diff_2 = abs(sample_5 - flipud(sample_5));
    diff_3 = abs(sample_5 - sample_5_rot);
    difference_5(i) = (sum(sum(diff_1))/256 + sum(sum(diff_2))/256)/2;
end

%Plot features
figure,

scatter(intensity_5, difference_5, 'Marker', 'x', 'MarkerEdgeColor', 'r');
hold on
scatter(intensity_1, difference_1, 'Marker', 'o', 'MarkerEdgeColor', 'b');

hold off
xlabel('Avg. Intensity');
ylabel('Avg. Difference');
legend('Dataset - 5','Dataset - 1');
fclose(fileID);




