function zernike_poly = zernike_fun(r,t,Anm,nflag)
% r是径向长度 t是幅角
%nflag是zernike类型
%0代表标准泽尼克系数，1代表条纹泽尼克系数 
%Anm代表泽尼克多项式系数 n行1列矩阵
%% 标准泽尼克系数
if(nflag==0)
                    ZRN_1=1;
                    ZRN_2=r*cos(t);
                    ZRN_3=r*sin(t);
                    ZRN_4=r.^2.*cos(2*t);
                    ZRN_5=2*r.^2-1;
                    ZRN_6=r.^2.*sin(2*t);

                    ZRN_7=r.^3.*cos(3*t);
                    ZRN_8=(3*r.^3-2*r).*cos(t);
                    ZRN_9=(3*r.^3-2*r).*sin(t);
                    ZRN_10=r.^3.*sin(3*t);

                    ZRN_11=r.^4.*cos(4*t);
                    ZRN_12=(4*r.^4-3*r.^2).*cos(2*t);
                    ZRN_13=6*r.^4-6*r.^2+1;
                    ZRN_14=(4*r.^4-3*r.^2).*sin(2*t);
                    ZRN_15=r.^4.*sin(4*t);

                    ZRN_16=r.^5.*cos(5*t);
                    ZRN_17=(5*r.^5-4*r.^3).*cos(3*t);
                    ZRN_18=(10*r.^5-12*r.^3+3*r).*cos(t);
                    ZRN_19=(10*r.^5-12*r.^3+3*r).*sin(t);
                    ZRN_20=(5*r.^5-4*r.^3).*sin(3*t);
                    ZRN_21=r.^5.*sin(5*t);

                    ZRN_22=r.^6.*cos(6*t);
                    ZRN_23=(6*r.^6-5*r.^4).*cos(4*t);%
                    ZRN_24=(15*r.^6-20*r.^4+6*r.^2).*cos(2*t);
                    ZRN_25=20*r.^6-30*r.^4+12*r.^2-1;
                    ZRN_26=(15*r.^6-20*r^4+6*r^2).*sin(2*t);
                    ZRN_27=(6*r.^6-5*r^4).*sin(4*t);
                    ZRN_28=r.^6.*sin(6*t);
                    ZRN_29=r.^7.*cos(7*t);
                    ZRN_30=(7*r.^7-6*r^5).*cos(5*t);
                    ZRN_31=(21*r.^7-30*r^5+10*r^3).*cos(3*t);
                    ZRN_32=(35*r.^7-60*r^5+30*r^3-4*r).*cos(t);
                    ZRN_33=(35*r.^7-60*r^5+30*r^3-4*r).*sin(t);
                    ZRN_34=(21*r.^7-30*r^5+10*r^3).*sin(3*t);
                    ZRN_35=(7*r.^7-6*r^5).*sin(5*t);
                    ZRN_36=r.^7.*sin(7*t);
       
%        zernike=[ZRN_4 ZRN_5 ZRN_6 ZRN_7 ZRN_8 ZRN_9 ZRN_10 ZRN_11 ZRN_12 ZRN_13 ZRN_14 ZRN_15...
%            ZRN_16 ZRN_17 ZRN_18 ZRN_19 ZRN_20 ZRN_21 ZRN_22 ZRN_23 ZRN_24 ZRN_25 ZRN_26 ZRN_27 ZRN_28 ZRN_29...
%            ZRN_30 ZRN_31 ZRN_32 ZRN_33 ZRN_34 ZRN_35 ZRN_36];
       zernike=[ZRN_5 ZRN_6 ZRN_7 ZRN_8 ZRN_9 ZRN_10 ZRN_11 ZRN_12 ZRN_13 ZRN_14 ZRN_15...
           ZRN_16 ZRN_17 ZRN_18 ZRN_19 ZRN_20 ZRN_21 ZRN_22 ZRN_23 ZRN_24 ZRN_25 ZRN_26 ZRN_27 ZRN_28 ZRN_29...
           ZRN_30 ZRN_31 ZRN_32 ZRN_33 ZRN_34 ZRN_35 ZRN_36];
     
end
    if(nflag==1)
                    ZRN_1=1;
                    ZRN_2=r*cos(t);
                    ZRN_3=r*sin(t);
                    ZRN_4=2*r.^2-1;
                    ZRN_5=r.^2.*cos(2*t);
                    ZRN_6=r.^2.*sin(2*t);

                    ZRN_7=(3*r.^3-2*r).*cos(t);
                    ZRN_8=(3*r.^3-2*r).*sin(t);
                    ZRN_9=6*r.^4-6*r.^2+1;
                    ZRN_10=r.^3.*cos(3*t);

                    ZRN_11=r.^3.*sin(3*t);
                    ZRN_12=(4*r.^4-3*r.^2).*cos(2*t);
                    ZRN_13=(4*r.^4-3*r.^2).*sin(2*t);
                    ZRN_14=(10*r.^5-12*r.^3+3*r).*cos(t);
                    ZRN_15=(10*r.^5-12*r.^3+3*r).*sin(t);

                    ZRN_16=20*r.^6-30*r.^4+12*r.^2-1;
                    ZRN_17=r.^4.*cos(4*t);
                    ZRN_18=r.^4.*sin(4*t);
                    ZRN_19=(5*r.^5-4*r.^3).*cos(3*t);
                    ZRN_20=(5*r.^5-4*r.^3).*sin(3*t);
                    ZRN_21=(15*r.^6-20*r.^4+6*r.^2).*cos(2*t);

                    ZRN_22=(15*r.^6-20*r.^4+6*r.^2).*sin(2*t);
                    ZRN_23=(35*r.^7-60*r^5+30*r.^3-4*r).*cos(t);
                    ZRN_24=(35*r.^7-60*r^5+30*r.^3-4*r).*sin(t);
                    ZRN_25=70*r.^8-140*r.^6+90*r.^4-20*r.^2+1;
                    ZRN_26=r.^5.*cos(5*t);
                    ZRN_27=r.^5.*sin(5*t);
                    ZRN_28=(6*r.^6-5*r.^4).*cos(4*t);
                    ZRN_29=(6*r.^6-5*r.^4).*sin(4*t);
                    ZRN_30=(21*r.^7-30*r.^5+10*r.^3).*cos(3*t);
                    ZRN_31=(21*r.^7-30*r.^5+10*r.^3).*sin(3*t);
                    ZRN_32=(56*r.^8-105*r.^6+60*r.^4-10*r.^2).*cos(2*t);
                    ZRN_33=(56*r.^8-105*r.^6+60*r.^4-10*r.^2).*sin(2*t);
                    ZRN_34=(126*r.^9-280*r.^7+210*r.^5-60*r.^3+5*r).*cos(t);
                    ZRN_35=(126*r.^9-280*r.^7+210*r.^5-60*r.^3+5*r).*sin(t);
                    ZRN_36=252*r.^10-630*r.^8+560*r.^6-210*r.^4+30*r.^2-1;
                    ZRN_37=924*r.^12-2772*r.^10+3150*r.^8-1680*r.^6+420*r.^4-42*r.^2+1;
       
%                     zernike=[ZRN_1 ZRN_2 ZRN_3 ZRN_4 ZRN_5 ZRN_6 ZRN_7 ZRN_8 ZRN_9 ZRN_10 ZRN_11 ZRN_12 ZRN_13 ZRN_14 ZRN_15...
%            ZRN_16 ZRN_17 ZRN_18 ZRN_19 ZRN_20 ZRN_21 ZRN_22 ZRN_23 ZRN_24 ZRN_25 ZRN_26 ZRN_27 ZRN_28 ZRN_29...
%            ZRN_30 ZRN_31 ZRN_32 ZRN_33 ZRN_34 ZRN_35 ZRN_36 ];%Z4-Z37
       
       zernike=[ZRN_5 ZRN_6 ZRN_7 ZRN_8 ZRN_9 ZRN_10 ZRN_11 ZRN_12 ZRN_13 ZRN_14 ZRN_15...
           ZRN_16 ZRN_17 ZRN_18 ZRN_19 ZRN_20 ZRN_21 ZRN_22 ZRN_23 ZRN_24 ZRN_25 ZRN_26 ZRN_27 ZRN_28 ZRN_29...
           ZRN_30 ZRN_31 ZRN_32 ZRN_33 ZRN_34 ZRN_35 ZRN_36];%Z4-Z36
       
    end
    if(nflag~=1 && nflag~=0)
      error('nflag must be 0 or 1.'); 
    end
    
    zernike_poly=zernike*Anm;
%% 
end


