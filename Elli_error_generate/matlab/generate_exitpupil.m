%% 调试日志  
% by 徐天晓
%2022.9.28
%主函数用作仿真加入异行光阑前后对PSF椭率测量的影响
%

%% 生成异形光瞳函数矩阵
clc;
clear;
close all;
samplingpoint_N=1024;
radius_ex=231;      %圆孔外半径 以4微米计算
radius_in=0;     %圆孔内半径
NKflag=1;
load('zernike_1_14_ss.mat');
zernike_poly_GXSS=zeros(samplingpoint_N,samplingpoint_N);

ExitPupil_or=zeros(samplingpoint_N,samplingpoint_N); 
x_ep=linspace(-samplingpoint_N/2+0.5,samplingpoint_N/2-0.5,samplingpoint_N);
y_ep=linspace(samplingpoint_N/2-0.5,-samplingpoint_N/2+0.5,samplingpoint_N);  
[x_ap,y_ap]=meshgrid(linspace(-samplingpoint_N/2+0.5,samplingpoint_N/2-0.5,samplingpoint_N));   
% D=((x_ap-a).^2+(y_ap-b).^2).^(1/2); %光瞳函数上一点到零点的距离
D=((x_ap).^2+(y_ap).^2).^(1/2); %光瞳函数上一点到零点的距离
distance_r=find(D>=radius_in & D<=radius_ex); 
ExitPupil=ExitPupil_or;
ExitPupil(distance_r)=1;    %设置孔半径范围内透射系数为1

% rectx=3;recty=40;%依据口径像素比例
% ExitPupil_rect=rect(x_ap/rectx).*rect(y_ap/recty);
% [xlab,ylab]=find(ExitPupil_rect==1);
% ExitPupil2=ExitPupil;
% ylab2=ylab+radius_ex-2;
% ExitPupil2(sub2ind(size(ExitPupil2),xlab,ylab2))=0;


[xlab,ylab]=find(ExitPupil==1);
ExitPupil2=ExitPupil_or;
ylab2=ylab-3;
% ExitPupil2(xlab,ylab2)=1;
ExitPupil2(sub2ind(size(ExitPupil2),xlab,ylab2))=1;
ExitPupil3=ExitPupil+ExitPupil2;

step=length(ExitPupil3);
for ii=1:step*step
    
    if ExitPupil3(ii)==1
        ExitPupil3(ii)=0;
    end
    
    if ExitPupil3(ii)==2
        ExitPupil3(ii)=1;
    end
end
ExitPupil_YX=ExitPupil2;  
% ExitPupil_original=ExitPupil_rect;  
% ExitPupil_original=ExitPupil3;  
figure('NumberTitle', 'off', 'Name', '衍射屏形状');
imagesc(ExitPupil_YX); %画衍射屏的形状 
colormap([0 0 0; 1 1 1]) %颜色以黑白区分 
axis image 
title('衍射屏形状');
% 
save('ExitPupil','ExitPupil');
%% 加入波像差
[polor_r,polor_t]=cartesian2polar(x_ep,y_ep,samplingpoint_N,samplingpoint_N,0,radius_ex);
Anm_GXSS=zernike_SS(5:36);
    for i_x=1:samplingpoint_N
        for i_y=1:samplingpoint_N
             r_temp=sqrt((x_ep(i_x)).^2+(y_ep(i_y)).^2); %计算各个像素到中心点的距离
             if (r_temp>=0 && r_temp<radius_ex)%干涉仪增加像差需要对整个环内加像差
                zernike_poly_GXSS(i_x,i_y)=zernike_fun(polor_r(i_x,i_y),polor_t(i_x,i_y),Anm_GXSS,NKflag);%调用zernike_fun函数计算zernike多项式 
             end
        end
    end

%     GXSS_aberrated=ExitPupil_GXSS.*exp(1i*2*pi*(zernike_poly_GXSS));
parallel_GXSS_aberrated_GT=ExitPupil_YX.*exp(1i*2*pi*(zernike_poly_GXSS));
% parallel_GXSS_aberrated_GT=ExitPupil_YX;
parallel_GXSS_aberrated_OR=ExitPupil.*exp(1i*2*pi*(zernike_poly_GXSS));
% parallel_GXSS_aberrated_OR=ExitPupil;
figure('NumberTitle', 'off', 'Name', '叠加波前图');
surf(x_ep,y_ep,zernike_poly_GXSS); %显示波像差图像
shading interp
axis equal
axis square
colormap jet;
colorbar
title('GXSS波前图')
view(0,90);%锁定投影视图

%% 求PSF
fft_parallel_GXSS_GT=fftshift(fft2(parallel_GXSS_aberrated_GT));    % 光学系统的广义光瞳函数傅里叶变换
PSF_parallel_GXSS_GT=fft_parallel_GXSS_GT.*conj(fft_parallel_GXSS_GT);
PSF_parallel_GXSS_GT=PSF_parallel_GXSS_GT./max(max(PSF_parallel_GXSS_GT));   

fft_parallel_GXSS_OR=fftshift(fft2(parallel_GXSS_aberrated_OR));    % 光学系统的广义光瞳函数傅里叶变换
PSF_parallel_GXSS_OR=fft_parallel_GXSS_OR.*conj(fft_parallel_GXSS_OR);
PSF_parallel_GXSS_OR=PSF_parallel_GXSS_OR./max(max(PSF_parallel_GXSS_OR));   

x_labe1=1:size(PSF_parallel_GXSS_GT,1);
y_labe1=1:size(PSF_parallel_GXSS_GT,2);
figure('NumberTitle', 'off', 'Name', '显示图像');
mesh(x_labe1,y_labe1,PSF_parallel_GXSS_GT);
axis equal
axis square
view(0,90)
colormap lines;
colorbar;
title('异形光瞳PSF');

% figure(3)
% x_labe1=1:size(PSF_parallel_GXSS_GT,1);
% y_labe1=1:size(PSF_parallel_GXSS_GT,2);
% mesh(x_labe1,y_labe1,PSF_parallel_GXSS_GT);
% view(0,90)
% axis square on
%% 计算椭率
[ellipticity_optmize(1),e1_optmize(1),e2_optmize(1)] = cac_ellipticity_funV2(PSF_parallel_GXSS_GT,1);%计算优化椭率参数
REE80_optmize(1) = GetEE80PSF(PSF_parallel_GXSS_GT);% 计算优化REE80

[ellipticity_optmize(2),e1_optmize(2),e2_optmize(2)] = cac_ellipticity_funV2(PSF_parallel_GXSS_OR,1);%计算优化椭率参数
REE80_optmize(2) = GetEE80PSF(parallel_GXSS_aberrated_OR);% 计算优化REE80

function z=rect(x)
x=abs(x);
z=(x<=1/2);
end

