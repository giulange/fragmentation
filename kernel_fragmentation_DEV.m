%% kernel fragmentation
%% pars
threshold           = 0.7;
mat_builtin         = false; % { true:colfilt.m , false:4-kernels-by-giuliano }
cuda_elapsed_time   = [910,260, 10870];% copy/paste from Nsight ==> [ 4-kernels, 3-kernels, giorgio.urso ]
RADIUS              = 5;
PATH                = '/home/giuliano/git/cuda/fragmentation';
% PATH              = '/Users/giuliano/Documents/MATLAB';
k_name{1}           = '-1-cumsum_horizontal.tif'    ;
k_name{2}           = '-2-sum_of_3_cols.tif'        ;
k_name{3}           = '-3-cumsum_vertical.tif'      ;
k_name{4}           = '-4-sum_of_3_rows.tif'        ;
%% ---input
% FIL_ROI           = fullfile(PATH,'data','ROI.tif');
% FIL_BIN           = fullfile(PATH,'data','BIN.tif');
FIL_ROI             = fullfile(PATH,'data','lodi1954_roi.tif');
FIL_BIN             = fullfile(PATH,'data','lodi1954.tif');
% FIL_ROI           = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif';
% FIL_BIN           = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954.tif';
% FIL_ROI             = fullfile(PATH,'data','imp_mosaic_char_2006_cropped_roi.tif');
% FIL_BIN             = fullfile(PATH,'data','imp_mosaic_char_2006_cropped.tif');
% FIL_ROI             = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2_roi.tif';
% FIL_BIN             = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2.tif';
% cuda intermediate files
FIL_k1              = fullfile(PATH,'data',k_name(1));
FIL_k2              = fullfile(PATH,'data',k_name(2));
FIL_k3              = fullfile(PATH,'data',k_name(3));
FIL_k4              = fullfile(PATH,'data',k_name(4));
FIL_FRAG            = fullfile(PATH,'data','FRAG-cuda.tif');
FIL_FRAGt           = fullfile(PATH,'data','FRAGt-cuda.tif');
FIL_FRAG_giorgio    = fullfile(PATH,'data','FRAGgiorgio-cuda.tif');
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
% info = geotiffinfo( FIL_BIN );
% ROI = logical(ones( info.Height, info.Width ));
% geotiffwrite(FIL_ROI,ROI,info.RefMatrix, ...
%     'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% set ROI as unsigned char
% info = geotiffinfo( FIL_ROI );
% ROI = geotiffread( FIL_ROI );
% ROI = logical( ROI );
% geotiffwrite(FIL_ROI,ROI,info.RefMatrix, ...
%     'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% CREATE new ROI given a BIN
% % FIL_ROI_new     = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif';
% FIL_ROI_new     = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2_roi.tif';
% info            = geotiffinfo(FIL_BIN);
% BIN             = geotiffread(FIL_BIN);
% ROI             = zeros(info.Height,info.Width);
% ROI(BIN>0)      = 1;
% geotiffwrite(FIL_ROI_new,ROI,info.RefMatrix, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% clip imperviousness grid for stress test & create ROI
% % see the /home/giuliano/git/cuda/fragmentation/data/README-giuly.txt in 
% !gdal_translate -srcwin 20000 40000 9958 9366 /home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006.tif /home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped.tif
% info = geotiffinfo( FIL_BIN );
% ROI = true(info.Height,info.Width);
% ROI(BIN==0)=false;
% geotiffwrite(FIL_ROI,ROI,info.RefMatrix, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% load BIN & ROI
BIN             = double(geotiffread(FIL_BIN));
ROI             = double(geotiffread(FIL_ROI));
% apply ROI by filtering BIN:
BIN = BIN.*ROI;
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
% % FRAG_ml = nlfilter( BIN, [mask_len, mask_len], myfun );
% % ...use this instead
FRAG_ml = colfilt( BIN ,[mask_len mask_len],'sliding',@sum);
FRAG_ml = FRAG_ml .* ROI;
myToc = round(toc*1000);
fprintf('%25s\t%6d [msec]\n\n','"colfilt" built-in func.',myToc)
end
%% DIFF :: MatLab - CUDA 4 kernel

if ~exist('FRAG_ml','var'), FRAG_ml = k4_ml; end
if ~mat_builtin
    algorithm = 'ml-4kernels';
elseif mat_builtin 
    algorithm = 'colfilt.m';
end

% print the speed-up
fprintf( '%25s\t%7.2f\n',['[',algorithm,' - cu] speed-up'],sum(myToc)/cuda_elapsed_time(1) )

FRAG_cuda       = geotiffread(FIL_FRAG);
DIFF            = FRAG_ml - FRAG_cuda;
fprintf('Number of pixels with wrong fragmentation:  %8d\n\n',sum(DIFF(:)~=0) )
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
%% preparation for transpose
% % % I need an array of size 2^[x,y] to use built-in CUDA function:
% % info = geotiffinfo('/home/giuliano/git/cuda/fragmentation/data/BIN.tif');
% % [BIN,R] = geotiffread('/home/giuliano/git/cuda/fragmentation/data/BIN.tif');
% % size_new = 2.^(floor(log2(size(BIN)))+1);
% % i = 1:prod(size_new);
% % BIN_2 = zeros(size_new);
% % BIN_2(i)=i;
% % R3 = maprasterref(R.worldFileMatrix,size_new);
% % % geotiffwrite('BIN-2.tif',BIN_2,R3,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)
% % % geotiffwrite('ROI-2.tif',double(ones(size(BIN_2))),R3,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)
% % geotiffwrite('BIN-2.tif',double(BIN),R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)
% % geotiffwrite('ROI-2.tif',double(ones(size(BIN))),R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)
%% DIFF :: MatLab - CUDA 3 kernel

fprintf( '%25s\t%7.2f\n',['[',algorithm,' - cu_t] speed-up'],sum(myToc)/cuda_elapsed_time(2) )

FRAG_cudat      = geotiffread( FIL_FRAGt );
DIFF            = FRAG_ml - FRAG_cudat;
fprintf('Number of pixels with wrong fragmentation:  %8d\n\n',sum(DIFF(:)~=0) )

% NOTE:
%   Remember that in CUDA you don't filter out pixels according to the kind
%   of fragmentation!!
% plot( sum(DIFF,1) )
%% DIFF :: CUDA 4 kernel - CUDA 3 kernel

fprintf( '%25s\t%7.2f\n','[cu / cu_t] speed-up',cuda_elapsed_time(1)/cuda_elapsed_time(2) )

DIFF            = FRAG_cuda - FRAG_cudat;
fprintf('Number of pixels with wrong fragmentation:  %8d\n\n',sum(DIFF(:)~=0) )

%% DIFF :: CUDA 3 kernel - CUDA Giorgio.Urso

fprintf( '%25s\t%7.2f\n','[cu-giorgio / cu_t] speed-up',cuda_elapsed_time(3)/cuda_elapsed_time(2) )
fprintf( '%25s\t%7.2f\n','[ml / cu-giorgio] speed-up', sum(myToc)/cuda_elapsed_time(3) )

FRAG_giorgio    = geotiffread( FIL_FRAG_giorgio );
DIFF            = FRAG_giorgio - FRAG_cudat;
fprintf('Number of pixels with wrong fragmentation:  %8d\n\n',sum(DIFF(:)~=0) )

%% explicit possible differences

DIFF            = FRAG_ml - FRAG_cudat;

[r,c]=find(DIFF~=0);

if ~isempty(r)
    fprintf('%6s%6s%14s%10s%10s\n','row','col','on-the-fly', 'matlab','cuda-T')
end
if length(r)>100, Nprints = 100; else Nprints=length(r); end
for ii = 1:Nprints
    rS = max(1,r(ii)-RADIUS);
    rE = min(HEIGHT,r(ii)+RADIUS);
    cS = max(1,c(ii)-RADIUS);
    cE = min(WIDTH,c(ii)+RADIUS);

    fprintf('%6d%6d%14d%10d%10d\n',r(ii),c(ii),sum(sum(BIN(rS:rE,cS:cE))), ...
            FRAG_ml(r(ii),c(ii)), FRAG_cudat(r(ii),c(ii)) )
end
if length(r)>100
    fprintf('...\n');
    rS = max(1,r(end)-RADIUS);
    rE = min(HEIGHT,r(end)+RADIUS);
    cS = max(1,c(end)-RADIUS);
    cE = min(WIDTH,c(end)+RADIUS);

    fprintf('%6d%6d%14d%10d%10d\n',r(end),c(end),sum(sum(BIN(rS:rE,cS:cE))), ...
            FRAG_ml(r(end),c(end)), FRAG_cudat(r(end),c(end)) )
end

%% step-by-step check

% [ROI]'
% K1 = geotiffread( fullfile(PATH,'data','-1-gtransform.tif') );
% sum(sum(double(K1)'-ROI))
% [BIN]'
% K2 = geotiffread( fullfile(PATH,'data','-2-gtransform.tif') );

% K3 = geotiffread( fullfile(PATH,'data','-3-Vcumsum.tif') );
% sum(sum(double(K3)-k1_ml))
% 
% K4 = geotiffread( fullfile(PATH,'data','-4-sum_of_3_LINES.tif') );
% sum(sum(double(K4)-k2_ml))
% DIFF = double(K4)-k2_ml;
% subplot(211),plot( sum(DIFF,1) )
% subplot(212),plot( sum(DIFF,2) )
% 
% K5 = geotiffread( fullfile(PATH,'data','-5-gtransform.tif') );
% 
% K6 = geotiffread( fullfile(PATH,'data','-6-Vcumsum.tif') );
% 
% K7 = geotiffread( fullfile(PATH,'data','FRAG-cuda.tif') );



