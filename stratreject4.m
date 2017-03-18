
function stratreject4(input_number)

%fileID11 = fopen(input_file,'r');
%A=fscanf(fileID11,'%f');

diary 

cvx_setup

datasets{1}='hay';
datasets{2}='bala';
datasets{3}='newconnect';
datasets{4}='iris';
datasets{5}='car';
datasets{6}='tissue';
datasets{7}='forest';
datasets{8}='heart';
datasets{9}='breast_cancer';
datasets{10}='guide3';
datasets{11}='ijcnn';
datasets{12}='bank';
datasets{13}='haber';
datasets{14}='pima';

lam1_vector=[0.01,0.1,1,10,100];
lam2_vector=[0.01,0.1,1,10,100];
d_vector=[1,2,3,4];
c_vector=[0.1,0.2,0.3,0.4,0.5];
dataset_vector=[6,5,7];

params=[0,0,0,0];

for datt=1:length(dataset_vector)
  for ccc=1:length(c_vector)
    for gg=1:length(d_vector)
      for ll=1:length(lam1_vector)
%	for ff=1:length(lam2_vector)
	       params=[params; dataset_vector(datt), c_vector(ccc),d_vector(gg), lam1_vector(ll)];
%	end
      end
    end
  end
end
params=params(2:end,:);

innum=str2num(input_number);
dataset_id=params(innum,1);
c=params(innum,2);
d=params(innum,3);
lam1=params(innum,4);
%lam2=params(innum,5);
lam2=lam2_vector;

data=csvread(datasets{dataset_id})';

turn=strcat('_c', num2str(c),'_data', num2str(dataset_id));

reject(data,c,d,lam1,lam2,strcat('genres',strcat(turn,'.txt')),strcat('gen_summary',strcat(turn,'.txt')),dataset_id)
diary off

end


function reject(x,c_vec,d_vec,lam1_vec,lam2_vec,file1,file2,dataset_id)

fileID=fopen(file1,'a+');
%fileID2=fopen(file2,'a+');

%fprintf(fileID,'\n %s \t %f \t %f \t','entrata:', num2str(c_vec),dataset_id);
%fprintf(fileID2,'\n %s \t %f \t %f \t','entrata:',num2str(c_vec),dataset_id);


%%%Diving data into predefined folds
for ii=1:5
    idx=randperm(size(x,2));
    cut=round(size(x,2)/5);
    val_fold{ii}=x(:,idx(1:cut));
    test_fold{ii}=x(:,idx(cut+1:2*cut));
    train_fold{ii}=x(:,idx(2*cut+1:end));
    
end

numremovedruns=0;
ssigma=[1.0,10, size(train_fold,2)];
% Running rejection
for i=1:length(c_vec);
%     min_val=1.5;
     
     for j=1:length(lam1_vec)
         for k=1:length(lam2_vec)
            for ii=1:length(d_vec)
                for kkk=1:length(d_vec)               
                  if d_vec(ii)<=3 && d_vec(kkk)<=3
                     disp('PARAMETERS')
                     disp(lam1_vec(j))
                     disp(lam2_vec(k))
                     disp(d_vec(ii))
                     disp(d_vec(kkk))

                    [train, val,test,eraserun]=reject_func(train_fold,val_fold,test_fold,c_vec(i),d_vec(ii),d_vec(kkk),lam1_vec(j),lam2_vec(k),1);
		    fprintf(fileID,'%s %f %s %f %s %f %s %f  %s %f %s %f %s %f %s %f %s %f %f %f %f %s %f %f %f %f %s %f  %f %f %f %s %f %f %f %f %s %f %f %f %f %s %f %f %f %f \n','dataset=',dataset_id,',cost=',c_vec(i),',eraserun=',eraserun,',d1=',d_vec(ii),',d2=',d_vec(kkk),',lam1=',lam1_vec(j),',lam2=',lam2_vec(k),',sigma=',0,',train=',train(1,:),'train_sd=',train(2,:) ,',val=',val(1,:),',val_sd=',val(2,:),',test=',test(1,:), ',test_sd=',test(2,:));

    
                     if eraserun==1
                         numremovedruns=numremovedruns+1;
                     end
                         
                    
%                     if min_val>val(1,2) && eraserun==0
%                         min_val=val(1,2);
%                         avg_train_min=train;
%                         avg_val_min=val;
%                         avg_test_min=test;
%                         c_min=c_vec(i);
%                         lam1_min=lam1_vec(j);
%                         lam2_min=lam2_vec(k);
%                         d_min1=d_vec(ii);
%                         d_min2=d_vec(kkk);
%                         sigma_min=1;
%                     end     
                  else 
                    for ss=1:length(ssigma)
                        disp('PARAMETERS')
                        disp(lam1_vec(j))
                        disp(lam2_vec(k))
                        disp(d_vec(ii))
                        disp(d_vec(kkk))

                       [train, val,test,eraserun]=reject_func(train_fold,val_fold,test_fold,c_vec(i),d_vec(ii),d_vec(kkk),lam1_vec(j),lam2_vec(k),ssigma(ss));
		       fprintf(fileID,'%s %f %s %f %s %f %s %f  %s %f %s %f %s %f %s %f %s %f %f %f %f %s %f %f %f %f %s %f  %f %f %f %s %f %f %f %f %s %f %f %f %f %s %f %f %f %f \n','dataset=',dataset_id,',cost=',c_vec(i),',eraserun=',eraserun,',d1=',d_vec(ii),',d2=',d_vec(kkk),',lam1=',lam1_vec(j),',lam2=',lam2_vec(k),',sigma=',ssigma(ss),',train=',train(1,:),'train_sd=',train(2,:) ,',val=',val(1,:),',val_sd=',val(2,:),',test=',test(1,:), ',test_sd=',test(2,:));
   
                         if eraserun==1
                            numremovedruns=numremovedruns+1;
                        end

%                        if min_val>val(1,2) && eraserun==0
%                            min_val=val(1,2);
%                            avg_train_min=train;
%                            avg_val_min=val;
%                            avg_test_min=test;
%                            c_min=c_vec(i);
%                            lam1_min=lam1_vec(j);
%                            lam2_min=lam2_vec(k);
%                            d_min1=d_vec(ii);
%                            d_min2=d_vec(kkk);
%                            sigma_min=ssigma(ss);

%                        end
                    end
                  end
                end 
            end
         end
     end
%     if min_val < 1.5
%        tabletxt(fileID,c_min,d_min1,d_min2,lam1_min,lam2_min,sigma_min,avg_train_min,avg_val_min,avg_test_min,numremovedruns)
%        tabletxt(fileID2,c_min,d_min1,d_min2,lam1_min,lam2_min,sigma_min,avg_train_min,avg_val_min,avg_test_min,numremovedruns)
%     end
   
end
fclose(fileID);
%fclose(fileID2);

end




function [avg_train, avg_val,avg_test,eraserun]=reject_func(train_fold,val_fold,test_fold,c,d1,d2,lam1,lam2,sigma)
% x is the input data that is a matrix of features by instances and row 1
% c is the cost
%reg_C is the regularization parameter

eraserun=0;

train_errors=ones(5,4);
val_errors=ones(5,4);
test_errors=ones(5,4);


%5 fold splits test val train
for i=1:5
    train=train_fold{i};
    val=val_fold{i};
    test=test_fold{i};

    train_label=train(1,:)';
    %if change this remember to change it for validation and test sets
    %too!!!!
    phi1=train(2:end,:);
    K1=kernel_matrix(phi1,d1,sigma);
    K2=kernel_matrix(phi1,d2,sigma);

    [alpha,beta1]=genalg_dual(K1,K2,lam1,lam2,train_label,c);

    %calculating errors
    %for rejection
    [t1,removeout1]=calc_err_genalg_dual(alpha,beta1,phi1,train_label,phi1,train_label, c,sigma,d1,d2);
    [t2,removeout2]=calc_err_genalg_dual(alpha,beta1,phi1,train_label,val(2:end,:) ,val(1,:)', c,sigma,d1,d2);
    [t3,removeout3]=calc_err_genalg_dual(alpha,beta1,phi1,train_label,test(2:end,:)  , test(1,:)', c,sigma,d1,d2);
    
    if removeout1==1 || removeout2==1 || removeout3==1 
        eraserun=1;
    end
    
     train_errors(i,:)=t1;
     val_errors(i,:)=t2;
     test_errors(i,:)=t3;

%calc_err_rej(u,b_u,w,b_w,test(2:end,:),test(2:end,:)  , test(1,:)',c);
end


avg_train=[mean(train_errors,1); std(train_errors,0,1)];
avg_val=[mean(val_errors,1); std(val_errors,0,1)];
avg_test=[mean(test_errors,1);std(test_errors,0,1)];


end




function [alpha,beta1]=genalg_dual(K1,K2,lam1,lam2,train_label,c)

    v1=(train_label~=1);
    v2=(train_label~=2);
    v3=(train_label~=3);
    v4=(train_label~=4);
    
    b1=1-v1;
    b2=1-v2;
    b3=1-v3;
    b4=1-v4;

    lams=lam1*lam2;
    cvx_begin
    variables alpha1(length(train_label)) alpha2(length(train_label)) alpha3(length(train_label)) alpha4(length(train_label))  beta1(length(train_label)) 
    maximize(-lam2*quad_form((-alpha1+b1.*(alpha1 + alpha2 + alpha3 +alpha4) )', K1) -lam2*quad_form((-alpha2+b2.*(alpha1+alpha2+alpha3+alpha4) )', K1) -lam2*quad_form((-alpha3+b3.*(alpha1+alpha2+alpha3+alpha4 ) )', K1)-lam2*quad_form((-alpha4+b4.*(alpha1+alpha2+alpha3+alpha4 ) )', K1) - 4*lam1*quad_form(c*beta1-0.5*v1.*alpha1-0.5*v2.*alpha2-0.5*v3.*alpha3-0.5*v4.*alpha4, K2)+ 8*lams*c*sum(beta1)+ 8*lams*sum(v1.*alpha1)+ 8*lams*sum(v2.*alpha2)+ 8*lams*sum(v3.*alpha3)+ 8*lams*sum(v4.*alpha4))
    subject to
    1 >=  v1.*alpha1+ v2.*alpha2 + v3.*alpha3 + v4.*alpha4+ c*beta1		   
    v1.*alpha1 >= 0
    v2.*alpha2 >= 0
    v3.*alpha3 >= 0
    v4.*alpha4 >= 0
    beta1 >= 0
    cvx_end
    
    alpha=[alpha1';alpha2';alpha3';alpha4'];


end




function out=poly_kernel(x,y,d,sigma)

    if d<=3
        out=((x')*y+1).^d;
    else
        out=exp(-(0.5/sigma)*norm(x-y)^2 );
    end


end


function K=kernel_matrix(train_data,d,sigma)
    K=zeros(size(train_data,2)); 
    for i=1:size(train_data,2)
         for j=1:size(train_data,2)
                  K(i,j)=poly_kernel(train_data(:,i),train_data(:,j),d,sigma);
         end
    end
    
end




function [minindex,throwout]=func_alpha(alpha,data,data_label,x,sigma,d1) %% w*x from alpha1 values 
    out=[0;0;0;0];
    for col=1:size(data,2) 
          out(1)= out(1)+ ( -alpha(1,col) + (data_label(col)==1)*(alpha(1,col)+alpha(2,col)+alpha(3,col)+alpha(4,col) ) ) *poly_kernel(data(:,col),x,d1,sigma);   
    end

    for col=1:size(data,2) 
          out(2)= out(2)+ ( -alpha(2,col) + (data_label(col)==2)*(alpha(1,col)+alpha(2,col)+alpha(3,col)+alpha(4,col)) ) *poly_kernel(data(:,col),x,d1,sigma);   
    end
    
    for col=1:size(data,2)  
          out(3)= out(3)+ ( -alpha(3,col) + (data_label(col)==3)*(alpha(1,col)+alpha(2,col)+alpha(3,col) +alpha(4,col)) ) *poly_kernel(data(:,col),x,d1,sigma);   
    end
    
    for col=1:size(data,2)  
          out(4)= out(4)+ ( -alpha(4,col) + (data_label(col)==4)*(alpha(1,col)+alpha(2,col)+alpha(3,col)+alpha(4,col) ) ) *poly_kernel(data(:,col),x,d1,sigma);   
    end
    
    
    if any(isnan(out)==1)
        throwout=1;
    else
        throwout=0;
    end
    
    [minvalue, minindex] = max(out);
    
end


function out=func_beta(alpha,beta1,c, data,data_label,x,sigma,d) %% u*x from alpha1 and beta1 values 
    out=0;
    for col=1:size(data,2)
          out= out+ ( c*beta1(col)-0.5*(data_label(col)~=1)*alpha(1,col)-0.5*(data_label(col)~=2)*alpha(2,col)-0.5*(data_label(col)~=3)*alpha(3,col)-0.5*(data_label(col)~=4)*alpha(4,col) )*poly_kernel(data(:,col),x,d,sigma);  
    end

end

function [errors,removerun]=calc_err_genalg_dual(alpha,beta1,train_data,train_label,test_data,test_data_label, c,sigma,d1,d2)

    rejection=[];
    learned_label=[];
    removerun=0;

    for values=1:size(test_data,2)
      
        [wx,throwout]=func_alpha(alpha,train_data,train_label,test_data(:,values),sigma,d1) ;
        if throwout==1
            break
        end
        
        ux=func_beta(alpha,beta1,c,train_data,train_label,test_data(:,values),sigma,d2) ;
        rejection= [rejection; ux];
        learned_label=[learned_label; wx];

    end
    
    if  throwout==1 || any(isnan(rejection)==1) 
        removerun=1;
        errors=[1,1,0,1];
    else
    
        err=sum(learned_label~=test_data_label)/length(learned_label);
        frac_rej=sum(rejection<0)/length(rejection);
        err_rej=c*sum(rejection<0);


        std_on_rej=0;
        for i=1:length(learned_label)
            if rejection(i)>=0 && learned_label(i)~=test_data_label(i)
                err_rej=err_rej+1;
                std_on_rej=std_on_rej+1;
            end
        end
        err_rej=err_rej/length(learned_label);
        std_on_rej=std_on_rej/length(learned_label);
        errors=[err,err_rej,frac_rej,std_on_rej];
    end
end



function tabletxt(fileID,c_min,d_min1,d_min2,lam1_min,lam2_min,sigma_min,avg_train_min,avg_val_min,avg_test_min,numremovedruns)

     fprintf(fileID,'%s \t %s \t  %4.5f \t %s \t %4.5f \t %s \t %4.5f \t %s \t %4.5f \t %s \t %4.5f \t %s \t %4.5f \t %s \t %4.5f \t','Rejection', 'cost',c_min,'d1',d_min1,'d2',d_min2,'lam1',lam1_min,'lam2',lam2_min,'sigma',sigma_min,'num runs removed=',numremovedruns);
     fprintf(fileID,'%s \t %4.3f%s%4.3f \t %4.3f%s%4.3f \t %4.3f%s%4.3f \t', 'std_err',avg_train_min(1,1),'+/-',avg_train_min(2,1),avg_val_min(1,1),'+/-',avg_val_min(2,1),avg_test_min(1,1), '+/-',avg_test_min(2,1));
     fprintf(fileID,'%s \t %4.3f%s%4.3f \t %4.3f%s%4.3f \t %4.3f%s%4.3f \t', 'rej_err',avg_train_min(1,2),'+/-',avg_train_min(2,2),avg_val_min(1,2),'+/-',avg_val_min(2,2),avg_test_min(1,2), '+/-',avg_test_min(2,2));
     fprintf(fileID,'%s \t %4.3f%s%4.3f \t %4.3f%s%4.3f \t %4.3f%s%4.3f \t', 'frac_rej',avg_train_min(1,3),'+/-',avg_train_min(2,3),avg_val_min(1,3),'+/-',avg_val_min(2,3),avg_test_min(1,3), '+/-',avg_test_min(2,3));
     fprintf(fileID,'%s \t %4.3f%s%4.3f \t %4.3f%s%4.3f \t %4.3f%s%4.3f \n', 'std_on_rej',avg_train_min(1,4),'+/-',avg_train_min(2,4),avg_val_min(1,4),'+/-',avg_val_min(2,4),avg_test_min(1,4), '+/-',avg_test_min(2,4));

end

