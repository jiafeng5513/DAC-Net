function ellipticity = cac_ellipticity_fun(PSF,k)
%计算PSF椭率
%% 给定光学系统参数
% FocInc=0.00074;%最小增量为0.74微米
FocInc=0.0004/k;%最小增量为0.4微米
%提取有效PSF  0.5″角半径
Eff_AngRia=0.5;%单位  角秒  选取PSF的有效角半径
FocalLength=28000;%GXSS的有效焦距 单位mm
Eff_Radius=0.0679;%FocalLength*tand(Eff_AngRia/3600);%PSF有效半径
Eff_Radius_num=floor(Eff_Radius./(FocInc));

%% PSF处理取80%能量集中区域
[x_max,y_max]=find(PSF==max(max(PSF)));
Eff_PSF=PSF((x_max-Eff_Radius_num):(x_max+Eff_Radius_num),(y_max-Eff_Radius_num):(y_max+Eff_Radius_num));%提取中间区域有效PSF
x_eff_psf=((-Eff_Radius_num):(Eff_Radius_num))*FocInc;%mm 有效PSF的横坐标
y_eff_psf=((-Eff_Radius_num):(Eff_Radius_num))*FocInc;%mm 有效PSF的纵坐标
for i_x=1:length(x_eff_psf)
    for i_y=1:length(y_eff_psf)
        if ((x_eff_psf(i_x)).^2+(y_eff_psf(i_y)).^2)>(Eff_Radius.^2)           
             Eff_PSF(i_x,i_y)=0;
        end
     end
end
dx=FocInc;%单位 mm
dy=FocInc;%单位 mm
%计算有效PSF的质心
F_PSF=sum(sum(Eff_PSF*dx)*dy);
num=1;
for i_x=1:length(x_eff_psf)
    for i_y=1:length(y_eff_psf)
        PSF_x_temp(num)=Eff_PSF(i_x,i_y).*x_eff_psf(i_x)*dx*dy;
        PSF_y_temp(num)=Eff_PSF(i_x,i_y).*y_eff_psf(i_y)*dx*dy;
        num=num+1;
     end
end
PSF_x_center=sum(PSF_x_temp)./F_PSF;%PSF的x向质心位置
PSF_y_center=sum(PSF_y_temp)./F_PSF;%PSF的y向质心位置
num=1;
for i_x=1:length(x_eff_psf)
     for i_y=1:length(y_eff_psf)
         Q_xx_temp(num)=Eff_PSF(i_x,i_y).*(x_eff_psf(i_x)-PSF_x_center).^2*dx*dy;
         Q_yy_temp(num)=Eff_PSF(i_x,i_y).*(y_eff_psf(i_y)-PSF_y_center).^2*dx*dy;
         Q_xy_temp(num)=Eff_PSF(i_x,i_y).*(x_eff_psf(i_x)-PSF_x_center).*(y_eff_psf(i_y)-PSF_y_center)*dx*dy;
         num=num+1;
      end
end
Qxx=sum(Q_xx_temp);
Qyy=sum(Q_yy_temp);
Qxy=sum(Q_xy_temp);

e1=(Qxx-Qyy)./(Qxx+Qyy);
e2=2*Qxy./(Qxx+Qyy);
ellipticity=sqrt(e1.^2+e2.^2);

end

