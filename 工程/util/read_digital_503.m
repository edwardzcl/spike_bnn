% %！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！%
% %響函旺廬算503忖憲鹿
digital_503.train_x=[];
digital_503.train_y=zeros(10,500);

for i=0:9
    for j=0:49 
        if j<10
           tmp=imread(fullfile('.\503忖憲鹿',[num2str(i),'00',num2str(j),'.bmp']));
        else
           tmp=imread(fullfile('.\503忖憲鹿',[num2str(i),'0',num2str(j),'.bmp']));
        end
        digital_503.train_x(:,:,50*i+j+1)=tmp;
        digital_503.train_y(i+1,50*i+j+1)=1;
    end
end

digital_503.train_x(find(digital_503.train_x==1))=2;
digital_503.train_x(find(digital_503.train_x==0))=1;
digital_503.train_x(find(digital_503.train_x==2))=0;


%！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！%

%！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！%
%響函旺廬算徭協吶忖憲鹿

digital_mine.train_x=[];
digital_mine.train_y=zeros(10,1000);


for i=0:9
    diroutput=dir(fullfile('C:\Users\xiguanyu\Desktop\matlab\retrain_mnist\bcnn_mnist\徭協吶忖憲鹿',num2str(i),'*.bmp'));
    for j=1:size(diroutput,1) 
        tmp=imread(fullfile('.\徭協吶忖憲鹿',num2str(i),diroutput(j).name));
        digital_mine.train_x(:,:,size(diroutput,1)*i+j)=tmp(:,:,1);
        digital_mine.train_y(i+1,size(diroutput,1)*i+j)=1;
    end
end

digital_mine.train_x(find(digital_mine.train_x<255))=2;
digital_mine.train_x(find(digital_mine.train_x>=255))=1;
digital_mine.train_x(find(digital_mine.train_x==1))=0;
digital_mine.train_x(find(digital_mine.train_x==2))=1;


digital_custom.train_x=cat(3,digital_503.train_x,digital_mine.train_x);
digital_custom.train_y=cat(2,digital_503.train_y,digital_mine.train_y);




        