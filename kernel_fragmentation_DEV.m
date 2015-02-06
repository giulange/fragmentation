%% kernel fragmentation
%% pars
% threshold       = 0.5;
mat_builtin         = true;
cuda_elapsed_time   = 920;% copy/paste from Nsight
RADIUS              = 5;
PATH                = '/home/giuliano/git/cuda/fragmentation';
% PATH              = '/Users/giuliano/Documents/MATLAB';
k_name{1}           = '-1-cumsum_horizontal.tif'    ;
k_name{2}           = '-2-sum_of_3_cols.tif'        ;
k_name{3}           = '-3-cumsum_vertical.tif'      ;
k_name{4}           = '-4-sum_of_3_rows.tif'        ;
%% ---input
% FIL_ROI         = fullfile(PATH,'data','ROI.tif');
% FIL_BIN         = fullfile(PATH,'data','BIN.tif');
FIL_BIN         = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954.tif';
FIL_ROI         = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif';
% cuda intermediate files
FIL_k1          = fullfile(PATH,'data',k_name(1));
FIL_k2          = fullfile(PATH,'data',k_name(2));
FIL_k3          = fullfile(PATH,'data',k_name(3));
FIL_k4          = fullfile(PATH,'data',k_name(4));
FIL_FRAG        = fullfile(PATH,'data','FRAG-cuda.tif');
%% create BIN
% info = geotiffinfo( FIL_ROI );
% 
% BIN = rand( info.Height, info.Width );
% 
% BIN(BIN>=threshold)=1;
% BIN(BIN<threshold)=0;
% BIN = logical(BIN);
% 
% geotiffwrite(FIL_BIN,BIN,info.RefMatrix, ...
%     'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% set ROI as all "ones"
% info = geotiffinfo( FIL_ROI );
% ROI = ones( info.Height, info.Width );
% geotiffwrite(FIL_ROI,ROI,info.RefMatrix, ...
%     'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% CREATE new ROI given a BIN
% FIL_ROI_new     = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif';
% info            = geotiffinfo(FIL_BIN);
% BIN             = geotiffread(FIL_BIN);
% ROI             = zeros(info.Height,info.Width);
% ROI(BIN>0)      = 1;
% geotiffwrite(FIL_ROI_new,ROI,info.RefMatrix, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% load BIN
BIN             = double(geotiffread(FIL_BIN));
%% set derived variables
buffer          = RADIUS+1;
mask_len        = RADIUS*2+1;
WIDTH           = size(BIN,2);
HEIGHT          = size(BIN,1);
gDx_k1_k2       = double(mod(WIDTH,mask_len)>0) + floor(WIDTH/mask_len);
gDy_k3_k4       = 1 + floor(HEIGHT / mask_len);
myToc           = zeros(4,1); % computing time [msec]
%% k1: compute cumsum along X [horizontal]
if ~mat_builtin
k1_ml       = zeros(size(BIN));

tic;
% first tile:
k1_ml(:,1)  = BIN(:,1);
for ii = 2:mask_len
    k1_ml(:,ii) = k1_ml(:,ii-1) + BIN(:,ii);
end
% all central tiles:
for tile = 0:gDx_k1_k2-2
    offset = tile*mask_len;
    k1_ml(:,offset+1)  = BIN(:,offset+1);
    for ii = 2:mask_len
        k1_ml(:,offset + ii) = k1_ml(:,offset+ii-1) + BIN(:,offset+ii);
    end
end
% last tile:
tile = gDx_k1_k2-1;
offset = tile*mask_len;
k1_ml(:,offset+1)  = BIN(:,offset+1);
for ii = 2:mask_len
    if ~ (offset+ii>WIDTH)
        k1_ml(:,offset + ii) = k1_ml(:,offset+ii-1) + BIN(:,offset+ii);
    end
end
myToc(1) = round(toc*1000);
fprintf('\n%25s\t%6d [msec]\n',k_name{1},myToc(1))
end
%% k2: compute sum of 3 cols
if ~mat_builtin
k2_ml       = zeros(HEIGHT,WIDTH);
clear latest_col latest_row

tic;
% ***first tile***
tile    = 0;
tid     = tile*mask_len;
%   *left*
for ii = 1:RADIUS
    k2_ml(:,tid+ii) = k1_ml(:,tid+ii+RADIUS);
end
%   *centre*
k2_ml(:,tid+RADIUS+1) = k1_ml(:,tid+mask_len);
%   *right*
for ii = RADIUS+2:mask_len
    k2_ml(:,tid+ii) = k1_ml(:,tid+ii+RADIUS) - k1_ml(:,tid+ii-RADIUS-1) + k1_ml(:,tid+mask_len);
end

% ***all central tiles***
for tile = 1:gDx_k1_k2-3
    tid = tile*mask_len;
    %   *left*
    for ii = 1:RADIUS
        k2_ml(:,tid+ii) = k1_ml(:,tid+ii+RADIUS) - k1_ml(:,tid+ii-RADIUS-1) + k1_ml(:,tid);
    end
    %   *centre*
    k2_ml(:,tid+RADIUS+1) = k1_ml(:,tid+mask_len);
    %   *right*
    for ii = RADIUS+2:mask_len
        k2_ml(:,tid+ii) = k1_ml(:,tid+ii+RADIUS) - k1_ml(:,tid+ii-RADIUS-1) + k1_ml(:,tid+mask_len);
    end
end
% tile before last one
tile = gDx_k1_k2-2;
tid = tile*mask_len;
%   *left*
for ii = 1:RADIUS
    k2_ml(:,tid+ii) = k1_ml(:,tid+ii+RADIUS) - k1_ml(:,tid+ii-RADIUS-1) + k1_ml(:,tid);
end
%   *centre*
k2_ml(:,tid+RADIUS+1) = k1_ml(:,tid+mask_len);
%   *right*
latest_col = WIDTH-tid;
for ii = RADIUS+2:mask_len
    k2_ml(:,tid+ii) = k1_ml(:,tid+min(ii+RADIUS,latest_col)) - k1_ml(:,tid+ii-RADIUS-1) + k1_ml(:,tid+mask_len);
end
% last tile:
tile = gDx_k1_k2-1;
tid = tile*mask_len;
latest_col = WIDTH-tid;
%   *left*
for ii = 1:RADIUS
    if tid+ii<=WIDTH
        k2_ml(:,tid+ii) = k1_ml(:,tid+min(ii+RADIUS,latest_col)) - k1_ml(:,tid+ii-RADIUS-1) + k1_ml(:,tid);
    end
end
%   *centre*
if tid+RADIUS+1<=WIDTH, k2_ml(:,tid+RADIUS+1) = k1_ml(:,tid+latest_col); end
%   *right*
for ii = RADIUS+2:mask_len
    if tid+ii<=WIDTH
        k2_ml(:,tid+ii) = k1_ml(:,tid+latest_col) - k1_ml(:,tid+ii-RADIUS-1);
    end
end
myToc(2) = round(toc*1000);
fprintf('%25s\t%6d [msec]\n',k_name{2},myToc(2))
end
%% k3: compute cumsum along X [horizontal]
if ~mat_builtin
k3_ml       = zeros(size(BIN));

tic;

% first tile:
k3_ml(1,:)  = k2_ml(1,:);
for ii = 2:mask_len
    k3_ml(ii,:) = k3_ml(ii-1,:) + k2_ml(ii,:);
end
% all central tiles:
for tile = 0:gDy_k3_k4-2
    offset = tile*mask_len;
    k3_ml(offset+1,:)  = k2_ml(offset+1,:);
    for ii = 2:mask_len
        k3_ml(offset + ii,:) = k3_ml(offset+ii-1,:) + k2_ml(offset+ii,:);
    end
end
% last tile:
tile = gDy_k3_k4-1;
offset = tile*mask_len;
k3_ml(offset+1,:)  = k2_ml(offset+1,:);
for ii = 2:mask_len
    if ~ (offset+ii>HEIGHT)
        k3_ml(offset + ii,:) = k3_ml(offset+ii-1,:) + k2_ml(offset+ii,:);
    end
end
myToc(3) = round(toc*1000);
fprintf('%25s\t%6d [msec]\n',k_name{3},myToc(3))
end
%% k4: compute sum of 3 rows
if ~mat_builtin
k4_ml       = zeros(HEIGHT,WIDTH);
clear latest_col latest_row

tic;
% ***first tile***
tile    = 0;
tid     = tile*mask_len;
%   *top*
for ii = 1:RADIUS
    k4_ml(tid+ii,:) = k3_ml(tid+ii+RADIUS,:);
end
%   *centre*
k4_ml(tid+RADIUS+1,:) = k3_ml(tid+mask_len,:);
%   *bottom*
for ii = RADIUS+2:mask_len
    k4_ml(tid+ii,:) = k3_ml(tid+ii+RADIUS,:) - k3_ml(tid+ii-RADIUS-1,:) + k3_ml(tid+mask_len,:);
end

% ***all central tiles***
for tile = 1:gDy_k3_k4-3
    tid = tile*mask_len;
    %   *top*
    for ii = 1:RADIUS
        k4_ml(tid+ii,:) = k3_ml(tid+ii+RADIUS,:) - k3_ml(tid+ii-RADIUS-1,:) + k3_ml(tid,:);
    end
    %   *centre*
    k4_ml(tid+RADIUS+1,:) = k3_ml(tid+mask_len,:);
    %   *bottom*
    for ii = RADIUS+2:mask_len
        k4_ml(tid+ii,:) = k3_ml(tid+ii+RADIUS,:) - k3_ml(tid+ii-RADIUS-1,:) + k3_ml(tid+mask_len,:);
    end
end
% tile before last one
tile = gDy_k3_k4-2;
tid = tile*mask_len;
%   *top*
for ii = 1:RADIUS
    k4_ml(tid+ii,:) = k3_ml(tid+ii+RADIUS,:) - k3_ml(tid+ii-RADIUS-1,:) + k3_ml(tid,:);
end
%   *centre*
k4_ml(tid+RADIUS+1,:) = k3_ml(tid+mask_len,:);
%   *bottom*
latest_row = HEIGHT-tid;
for ii = RADIUS+2:mask_len
    k4_ml(tid+ii,:) = k3_ml(tid+min(ii+RADIUS,latest_row),:) - k3_ml(tid+ii-RADIUS-1,:) + k3_ml(tid+mask_len,:);
end
% last tile:
tile = gDy_k3_k4-1;
tid = tile*mask_len;
latest_row = HEIGHT-tid;
%   *top*
for ii = 1:RADIUS
    if tid+ii<=HEIGHT
        k4_ml(tid+ii,:) = k3_ml(tid+min(ii+RADIUS,latest_row),:) - k3_ml(tid+ii-RADIUS-1,:) + k3_ml(tid,:);
    end
end
%   *centre*
if tid+RADIUS+1<=HEIGHT, k4_ml(tid+RADIUS+1,:) = k3_ml(tid+latest_row,:); end
%   *bottom*
for ii = RADIUS+2:mask_len
    if tid+ii<=HEIGHT
        k4_ml(tid+ii,:) = k3_ml(tid+latest_row,:) - k3_ml(tid+ii-RADIUS-1,:);
    end
end
myToc(4) = round(toc*1000);
fprintf('%25s\t%6d [msec]\n',k_name{4},myToc(4))
fprintf('%s\n',repmat('_',50,1))
fprintf('%25s\t%6d [msec]\n\n\n','Total time',sum(myToc))
end
%% MatLab FRAGMENTATION
if mat_builtin
myfun = @(in) sum(in(:));
tic
% % To much time consuming!
% % FRAG_ml = nlfilter(BIN, [mask_len, mask_len], myfun );
% % ...use this instead
FRAG_ml = colfilt(BIN,[mask_len mask_len],'sliding',@sum);
myToc = round(toc*1000);
fprintf('%25s\t%6d [msec]\n','"colfilt" built-in func.',myToc)
end
%% DIFF :: MatLab - CUDA
% print the speed-up
fprintf( '%25s\t%4.2f [msec]\n','speed-up',sum(myToc)/cuda_elapsed_time )

if ~exist('FRAG_ml','var'), FRAG_ml = k4_ml; end
FRAG_cuda       = geotiffread(FIL_FRAG);
DIFF            = FRAG_ml - FRAG_cuda;
fprintf('Number of pixels with wrong fragmentation:%8d\n\n',length(find(DIFF(:))))
%% check temporary arrays
% % k1_cuda = geotiffread(FIL_k1);
% % sum( k1_cuda(:)-k1_ml(:) )
% % % DIFF_k1 = k1_cuda-k1_ml;
% % 
% % k2_cuda = geotiffread(FIL_k2);
% % sum( k2_cuda(:)-k2_ml(:) )
% % % DIFF_k2 = k2_cuda-k2_ml;
% % 
% % k3_cuda = geotiffread(FIL_k3);
% % sum( k3_cuda(:)-k3_ml(:) )
% % % DIFF_k3 = k3_cuda-k3_ml;
% % 
% % k4_cuda = geotiffread(FIL_k4);
% % sum( k4_cuda(:)-k4_ml(:) )
% % % DIFF_k4 = k4_cuda-k4_ml;
% % 
%% rural fragmentation
% % % write in global F only if the pixel is ONE
% % Fr = F;
% % Fr(BIN==1)=0;
%% urban fragmentation
% % % write in global F only if the pixel is ZERO
% % max_val_in_mask = (RADIUS*2+1)^2;
% % Fu = max_val_in_mask-F;
% % Fu(BIN==0)=0;
