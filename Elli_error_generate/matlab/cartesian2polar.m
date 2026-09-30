function [r,t]=cartesian2polar(x_ep,y_ep,samplingpoint_x,samplingpoint_y,radius_in,radius_ex)
   r=zeros(samplingpoint_x,samplingpoint_y);
   t=zeros(samplingpoint_x,samplingpoint_y);
    for i_x=1:samplingpoint_x
        for i_y=1:samplingpoint_y
            
            r_temp=sqrt((x_ep(i_x)).^2+(y_ep(i_y)).^2); %计算各个像素到中心点的距离
%             r_temp_cac(i_x,i_y)=r_temp;%放入矩阵r_temp_cac 以供查验
            if (r_temp>=radius_in && r_temp<radius_ex)%光瞳函数加像差 即通光部分
                if (x_ep(i_x)==0 && y_ep(i_y)==0)
                    r(i_x,i_y)=0;
                    t(i_x,i_y)=0;
%                     cac_num=cac_num+1; 
                end
                if (x_ep(i_x)>=0 && y_ep(i_y)~=0)%正弦函数在一四象限具有单调性
                    r(i_x,i_y)=r_temp./radius_ex;
                    t(i_x,i_y)=asin(y_ep(i_y)./r_temp);
                end
                if (x_ep(i_x)>=0 && y_ep(i_y)==0)%正弦函数在一四象限具有单调性
                    r(i_x,i_y)=r_temp./radius_ex;
                    t(i_x,i_y)=0;
                end
                if (x_ep(i_x)<0 && y_ep(i_y)>=0)
                    r(i_x,i_y)=r_temp./radius_ex;
                    t(i_x,i_y)=acos(x_ep(i_x)./r_temp);%余弦函数在二象限
                end
                if (x_ep(i_x)<0 && y_ep(i_y)<0)
                    r(i_x,i_y)=r_temp./radius_ex;
                    t(i_x,i_y)=-acos(x_ep(i_x)./r_temp);%余弦函数在三象限
                end
            end
        end
    end
    r=r';
    t=t';
end

