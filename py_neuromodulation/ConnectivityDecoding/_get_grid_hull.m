addpath('C:\code\wjn_toolbox');
addpath(genpath('C:\code\leaddbs'));
addpath(genpath('C:\code\spm12'));


%%
ctx = wjn_mni_cortex();
downsample_ctx=ctx.vertices(1:20:end,:); %downsample by 10

save("downsampled_cortex.mat", "downsample_ctx")

figure;
scatter3(downsample_ctx(:,1), downsample_ctx(:,2), downsample_ctx(:,3), 'filled');
title('3D Scatter Plot Example');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
grid on;



PATH_OUT = "D:\Connectome_RMAP_OUT\ROIs";

for a =1:size(downsample_ctx,1)
    disp(a)
    roiname  = fullfile(PATH_OUT, strcat('ROI-', string(a), '.nii'));
    mni = [downsample_ctx(a, 1) downsample_ctx(a, 2) downsample_ctx(a, 3)];
    wjn_spherical_roi(roiname,mni,4);
end





