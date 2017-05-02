clc;clear;close all;
dataDir = '/usr3/graduate/mughani/EC591/tf_unet-master/bp_ang90_snr20/test/';%fullfile('data', '291');
mkdir('bp_ang90_snr20_test');
count = 0;
f_lst = dir(fullfile(dataDir, '*_mask.tif'));
for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
%     disp(f_info.name);
    if f_info.name == '.'
        continue;
    end

    f = strsplit(f_info.name,'_mask.tif');
    f_name = sprintf('%s.tif',f{1,1});
    f_path = fullfile(dataDir,f_name);
    img_2 = imread(f_path);
    img_2 = img_2(:,:,1);
    img_2 = im2double(img_2);

    g_path = fullfile(dataDir,f_info.name);
    disp(g_path);
    img_raw = imread(g_path);
    img_raw = img_raw(:,:,1);
    img_raw = im2double(img_raw);

    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);

    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    img_2 = img_2(1:height-mod(height,12),1:width-mod(width,12),:);

    img_raw = img_raw(34:289,34:289);
    img_2 = img_2(34:289,34:289);
    size(img_raw)
    img_size = size(img_raw);

%     img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
%     img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
%     img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
%
    patch_name = sprintf('bp_ang90_snr20_test/%d',count);

    save(patch_name, 'img_raw');
    save(sprintf('%s_2', patch_name), 'img_2');
%     save(sprintf('%s_3', patch_name), 'img_3');
%     save(sprintf('%s_4', patch_name), 'img_4');

    count = count + 1;
    display(count);
end
