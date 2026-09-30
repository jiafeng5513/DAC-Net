function Anm=RandNum(n,RMS_Level,nflag)
%% 调试日志  
% by 徐天晓
%用作主函数sim_main、sim_mainV2的随机泽尼克系数生成
%n代表zernike详述 RMS_Level代表RMS 
%nflag 泽尼克多项式类型 
%1代表条纹泽尼克多项式
%0代表标准泽尼克多项式
%%
%     Coff_ZRN=[1,4,4,3,6,6,8,8,8,8,5,10,10,10,10,12,12,12,12,12,12,7,14,14,14,14,14,14,16,16,16,16,16,16,16,16,9];%CodeV中标准Zernike系数的常数项
%     Coff_ZFC_Z4_Z36=[3,6,6,8,8,5,8,8,10,10,12,12,7,10,10,12,12,14,14,16,16,9,12,12,14,14,16,16,18,18,20,20,11,13];
%     Coff_ZFC_Z1_Z36=[1,4,4,3,6,6,8,8,5,8,8,10,10,12,12,7,10,10,12,12,14,14,16,16,9,12,12,14,14,16,16,18,18,20,20,11];
      Coff_ZFC_Z5_Z36=[6,6,8,8,5,8,8,10,10,12,12,7,10,10,12,12,14,14,16,16,9,12,12,14,14,16,16,18,18,20,20,11];
%     Coff_ZRN_Z4_Z36=[3,6,6,8,8,8,8,5,10,10,10,10,12,12,12,12,12,12,7,14,14,14,14,14,14,16,16,16,16,16,16,16,16,9];%CodeV中标准Zernike系数的常数项
    % Coff_ZFR=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    Coff_ZSC = ones(n,1);%CodeV中条纹Zernike系数的常数项
    if(nflag==0) %0代表标准泽尼克多项式
       RandNum=(rand(n,1)-0.5)*2;
       Sumrandnum=0;
       for sum_i=1:length(RandNum)
          Sumrandnum=Sumrandnum+RandNum(sum_i).^2/Coff_ZSC(sum_i);
       end
    %    Anm=Anm.*RandNum./sqrt(Sumrandnum).*RMS_Level;
       Anm=RandNum/sqrt(Sumrandnum).*RMS_Level;
    end

    if(nflag==1) %1代表条纹泽尼克多项式
       RandNum=(rand(n,1)-0.5)*2;
       Sumrandnum=0;
       for sum_i=1:length(RandNum)
          Sumrandnum=Sumrandnum+RandNum(sum_i).^2*Coff_ZSC(sum_i);
       end
       Anm=RandNum./sqrt(Sumrandnum).*RMS_Level;
       Anm=Anm.*sqrt(Coff_ZFC_Z5_Z36');
    end
    
    if(nflag~=1 && nflag~=0)
      error('nflag must be 0 or 1.'); 
    end
    
end

