%% 调试日志  
% by 徐天晓
%2026.9.23
%本程序作用是生成不同的GXSS的波像差
%通过更改循环参数sim_total_num可以定义n个样本
%程序运行后的数据保存在ansmat和initial


clc;
clear;
close all;
tic;
%% 导入初始标准圆 进行预处理
load('ExitPupil.mat');%由光瞳导入图像
image=ExitPupil;
[image_x,image_y]=size(image);
if mod(image_x,2)==0
%     samplingpoint_N=image_x/2-1;      %采样点数为偶数
    samplingpoint_N=image_x-1;      %采样点数为偶数
else
    samplingpoint_N=(image_x-1)/2;    %采样点数为奇数
end
ExitPupil_original=image;  
figure('NumberTitle', 'off', 'Name', '衍射屏原始形状');
imagesc(ExitPupil_original); %画衍射屏的形状 
colormap([0 0 0; 1 1 1]) %颜色以黑白区分 
axis image 
title('衍射屏原始形状');

%% 仿真参数设置
Anm_parallel_num=32;       %Z4-Z36
Anm_interferometer_num=32; %Z4-Z36
NKflag=1;%0代表标准泽尼克多项式,1代表条纹泽尼克多项式
RMS_parallel_Level=1/14;%平行光管波像差RMS
RMS_GXSS_Level=1/14;%干涉仪波像差RMS


%% 定义循环参数
sim_total_num=2;%通过更改循环参数sim_total_num可以定义n个样本

% misalignment=zeros(sim_total_num,5);
% test_zernike_RMS=zeros(sim_total_num,3);
ansmat(sim_total_num).ellipticity_parallel_GXSS = [];
ansmat(sim_total_num).PSF = [];
%% 光瞳参数设置
radius_ex=231;    %圆孔外半径 
radius_in=0;     %圆孔内半径
% radius_k=radius_in/radius_ex;     %内半径与外半径之比
zernike_poly_parallel=zeros(samplingpoint_N,samplingpoint_N);
zernike_poly_GXSS=zeros(samplingpoint_N,samplingpoint_N);
zernike_poly_parallel_GXSS=zeros(samplingpoint_N,samplingpoint_N);
r_temp_cac=zeros(samplingpoint_N,samplingpoint_N);

%% 平行光管广义光瞳函数
ExitPupil_parallel=zeros(samplingpoint_N,samplingpoint_N); 
x_ep=linspace(-samplingpoint_N/2+0.5,samplingpoint_N/2-0.5,samplingpoint_N);
y_ep=linspace(samplingpoint_N/2-0.5,-samplingpoint_N/2+0.5,samplingpoint_N);  
[x_ap,y_ap]=meshgrid(linspace(-samplingpoint_N/2+0.5,samplingpoint_N/2-0.5,samplingpoint_N));   
% D=((x_ap-a).^2+(y_ap-b).^2).^(1/2); %光瞳函数上一点到零点的距离
D=((x_ap).^2+(y_ap).^2).^(1/2); %光瞳函数上一点到零点的距离
distance_r1=find(D>=radius_in & D<=radius_ex); 
ExitPupil_parallel(distance_r1)=1;    %设置孔半径范围内透射系数为1

ExitPupil_original=ExitPupil_parallel;  
figure('NumberTitle', 'off', 'Name', '衍射屏原始形状');
imagesc(ExitPupil_original); %画衍射屏的形状 
colormap([0 0 0; 1 1 1]) %颜色以黑白区分 
axis image 
title('平行光管光瞳形状');
%% GXSS广义光瞳函数

ExitPupil_GXSS=zeros(samplingpoint_N,samplingpoint_N); 
distance_r2=find(D>=0 & D<=radius_ex); 
ExitPupil_GXSS(distance_r2)=1;    %设置孔半径范围内透射系数为1

ExitPupil_original=ExitPupil_GXSS;  
figure('NumberTitle', 'off', 'Name', '衍射屏原始形状');
imagesc(ExitPupil_original); %画衍射屏的形状 
colormap([0 0 0; 1 1 1]) %颜色以黑白区分 
axis image 
title('GXSS光瞳形状');

[polor_r,polor_t]=cartesian2polar(x_ep,y_ep,samplingpoint_N,samplingpoint_N,0,radius_ex);

%% 平行光管波像差
    Anm_parallel = RandNum(Anm_parallel_num,RMS_parallel_Level,NKflag); %生成RMS固定的zernike多项式随机系数
        for i_x=1:samplingpoint_N
            for i_y=1:samplingpoint_N
                 r_temp=sqrt((x_ep(i_x)).^2+(y_ep(i_y)).^2); %计算各个像素到中心点的距离
                 if (r_temp>=0 && r_temp<radius_ex)%光瞳函数加像差 
                    zernike_poly_parallel(i_x,i_y)=zernike_fun(polor_r(i_x,i_y),polor_t(i_x,i_y),Anm_parallel,NKflag);%调用zernike_fun函数计算zernike多项式
                 end
            end
        end

    parallel_aberrated=ExitPupil_parallel.*exp(1i*2*pi*zernike_poly_parallel);
    
    initial.parallel_zernike=Anm_parallel;
    initial.parallel_poly_zernike=zernike_poly_parallel;
    initial.parallel_aberrated=parallel_aberrated;
%     cac_zernike=(nonzeros(zernike_poly_parallel))';%取矩阵中所有非零元素组成一维矩阵
%     test_zernike_RMS(step_num,1)=rms(cac_zernike);

%% GXSS波像差
    Anm_GXSS = RandNum(Anm_interferometer_num,RMS_GXSS_Level,NKflag); %生成RMS固定的zernike多项式随机系数
        for i_x=1:samplingpoint_N
            for i_y=1:samplingpoint_N
                 r_temp=sqrt((x_ep(i_x)).^2+(y_ep(i_y)).^2); %计算各个像素到中心点的距离
                 if (r_temp>=0 && r_temp<radius_ex)%干涉仪增加像差需要对整个环内加像差
                    zernike_poly_GXSS(i_x,i_y)=zernike_fun(polor_r(i_x,i_y),polor_t(i_x,i_y),Anm_GXSS,NKflag);%调用zernike_fun函数计算zernike多项式 
                 end
            end
        end
  
    GXSS_aberrated=ExitPupil_GXSS.*exp(1i*2*pi*(zernike_poly_GXSS));
    initial.GXSS_zernike = Anm_GXSS;
    initial.GXSS_poly_zernike=zernike_poly_GXSS;
    initial.GXSS_aberrated=GXSS_aberrated;
    
    
%     cac_zernike=(nonzeros(zernike_poly_GXSS))';%取矩阵中所有非零元素组成一维矩阵
%     test_zernike_RMS(step_num,2)=rms(cac_zernike);
%% 光学系统理论波像差
    Anm_parallel_GXSS = Anm_parallel+Anm_GXSS;
        for i_x=1:samplingpoint_N
            for i_y=1:samplingpoint_N
                 r_temp=sqrt((x_ep(i_x)).^2+(y_ep(i_y)).^2); %计算各个像素到中心点的距离
                 if (r_temp>=0 && r_temp<radius_ex)%干涉仪增加像差需要对整个环内加像差
                    zernike_poly_parallel_GXSS(i_x,i_y)=zernike_fun(polor_r(i_x,i_y),polor_t(i_x,i_y),Anm_parallel_GXSS,NKflag);%调用zernike_fun函数计算zernike多项式 
                 end
            end
        end
       parallel_GXSS_aberrated=ExitPupil_GXSS.*exp(1i*2*pi*(zernike_poly_parallel_GXSS));
       initial.parallel_GXSS_poly_zernike = zernike_poly_parallel_GXSS;
       initial.parallel_GXSS_aberrated = parallel_GXSS_aberrated;
  
        fft_parallel_GXSS=fftshift(fft2(parallel_GXSS_aberrated));    % 含有误差的光学系统的广义光瞳函数傅里叶变换
        PSF_parallel_GXSS=fft_parallel_GXSS.*conj(fft_parallel_GXSS);
        PSF_parallel_GXSS=PSF_parallel_GXSS./max(max(PSF_parallel_GXSS));  %归一化
        initial.parallel_GXSS_PSF=PSF_parallel_GXSS;
        
for step_num=1:sim_total_num %开始循环
%% 光学系统带误差的波像差
    Deta_error = RandNum(Anm_interferometer_num,RMS_GXSS_Level,NKflag);
    Anm_parallel_GXSS = Anm_parallel+Anm_GXSS+Deta_error;
        for i_x=1:samplingpoint_N
            for i_y=1:samplingpoint_N
                 r_temp=sqrt((x_ep(i_x)).^2+(y_ep(i_y)).^2); %计算各个像素到中心点的距离
                 if (r_temp>=0 && r_temp<radius_ex)%干涉仪增加像差需要对整个环内加像差
                    zernike_poly_parallel_GXSS(i_x,i_y)=zernike_fun(polor_r(i_x,i_y),polor_t(i_x,i_y),Anm_parallel_GXSS,NKflag);%调用zernike_fun函数计算zernike多项式 
                 end
            end
        end
 
    parallel_GXSS_aberrated=ExitPupil_parallel.*exp(1i*2*pi*(zernike_poly_parallel_GXSS));
    ansmat(step_num).Deta_error = Deta_error;
%     cac_zernike=(nonzeros(zernike_poly_parallel_GXSS))';%取矩阵中所有非零元素组成一维矩阵
%     test_zernike_RMS(step_num,3)=rms(cac_zernike); 
    
%% 求PSF
%     fft_parallel=fftshift(fft2(parallel_aberrated));     %平行光管的广义光瞳函数傅里叶变换
%     PSF_parallel=fft_parallel.*conj(fft_parallel);
%     % PSF_BH=(abs(B)).^2;
%     PSF_parallel=PSF_parallel./max(max(PSF_parallel));  %归一化
  
    fft_parallel_GXSS=fftshift(fft2(parallel_GXSS_aberrated));    % 含有误差的光学系统的广义光瞳函数傅里叶变换
    PSF_parallel_GXSS=fft_parallel_GXSS.*conj(fft_parallel_GXSS);
    PSF_parallel_GXSS=PSF_parallel_GXSS./max(max(PSF_parallel_GXSS));  %归一化
%% 经系统退化成像
%     ima_g1=conv2(image,PSF_parallel,'same');%经光学系统成像  
%     ima_g2=conv2(ima_g1,PSF_GXSS,'same');%经光学系统成像  
%     ima_g1=conv2(image,PSF_parallel_GXSS,'same');%经光学系统成像  

%% 计算椭率
%     PSF_pic_parallel_GXSS=ima_g1./max(max(ima_g1));
    % PSF_recover_obstruct=ima_LR_obstruct./max(max(ima_LR_obstruct));
%     PSF_recover_if=ima_LR_if./max(max(ima_LR_if));
    % PSF_recover_expand=ima_LR_if_expand./max(max(ima_LR_if_expand));

    ellipticity_parallel_GXSS = cac_ellipticity_fun(PSF_parallel_GXSS,1);%原始图像椭率
    
    % ellipticity_recover_obstruct = cac_ellipticity_fun(PSF_recover_obstruct)%去遮拦PSF恢复后椭率
%     ellipticity_recover_if = cac_ellipticity_fun(PSF_recover_if,1);%干涉仪PSF恢复后椭率
    % ellipticity_recover_expand = cac_ellipticity_fun(PSF_recover_expand,2)%插值后的椭率

    ansmat(step_num).ellipticity_parallel_GXSS=ellipticity_parallel_GXSS;
    ansmat(step_num).PSF=PSF_parallel_GXSS;

end

fft_GXSS=fftshift(fft2(GXSS_aberrated));    % GXSS的广义光瞳函数傅里叶变换
PSF_GXSS=fft_GXSS.*conj(fft_GXSS);
PSF_GXSS=PSF_GXSS./max(max(PSF_GXSS));  %归一化 
ellipticity_GXSS = cac_ellipticity_fun(PSF_GXSS,1);%光学系统成像后椭率
initial.ellipticity_GXSS = ellipticity_GXSS;


save('ansmat','ansmat');
save('initial','initial');
%% 画图
% figure('NumberTitle', 'off', 'Name', '计算结果');
% plot(ans_mat(:,1),'b-s','linewidth',1.5);
% hold on
% 
% plot(ans_mat(:,2),'r-*','linewidth',1.5);
% hold on
% 
% plot(ans_mat(:,3),'g-o','linewidth',1.5);
% hold off
% grid on
% legend('e parallel and GXSS','e GXSS','e LR')
% title('LR算法遮拦恢复');
% 
% GXSS_PG=ans_mat(:,2)-ans_mat(:,1);
% GXSS_LR=ans_mat(:,2)-ans_mat(:,3);
% mean_abs_GXSS_PG=mean(abs(GXSS_PG))
% mean_abs_GXSS_LR=mean(abs(GXSS_LR))
% figure('NumberTitle', 'off', 'Name', '分析结果');
% % plot([1,sim_total_num],[0,0],'k-','linewidth',1);
% % plot(ans_mat(:,2),'k-','linewidth',1);
% % hold on
% plot(GXSS_PG,'r-','linewidth',1);
% hold on
% plot(GXSS_LR,'b-','linewidth',1);
% hold off
% % bar([GXSS_PG GXSS_LR]);
% legend('e GXSS-PG','e GXSS-LR');
% title('分析');
% 
% cac_data=abs(GXSS_PG)-abs(GXSS_LR);
% data1=length(find(cac_data>0))
% data2=500-data1;
% % y=[data1;data2];
% % figure('NumberTitle', 'off', 'Name', '分析结果');
% % bar(y);
% % legend('e GXSS-PG','e GXSS-LR');
% % title('分析');

toc;
beep;
