clear;
clc;

load ff_parameter

% input_data=zeros(512,1);
% 
% for i=1:4
%     filename=['output',num2str(i-1),'.txt'];
%     fid{i}=fopen(filename,'r');
% end
% 
% for i=1:numel(fid)
%     line = 1;
%     while feof(fid{i}) == 0   
%         str_line{i,line} = fgetl(fid{i});
%         data_line{i}(line) = bin2dec(str_line{i,line}(27:36))+1;
%         line = line+1;
%     end
%     input_data(data_line{i})=1;
% end


for k=1:10
    ffBW{k}=zeros(4,4,32);
    ffBW{k}(:)=ff_parameter.ffBW(k,:);
    
    core1_map=ffBW{k}(1:2,1:2,:);
    core2_map=ffBW{k}(1:2,3:4,:);
    core3_map=ffBW{k}(3:4,1:2,:);
    core4_map=ffBW{k}(3:4,3:4,:);

    core1_map=permute(core1_map,[2,1,3]);
    core2_map=permute(core2_map,[2,1,3]);
    core3_map=permute(core3_map,[2,1,3]);
    core4_map=permute(core4_map,[2,1,3]);

    ff_transfer.BW{k}(1,1:128)=core1_map(:);
    ff_transfer.BW{k}(1,129:256)=core2_map(:);
    ff_transfer.BW{k}(1,257:384)=core3_map(:);
    ff_transfer.BW{k}(1,385:512)=core4_map(:);
    
    ff_transfer.Bb{k}=ff_parameter.ffBb(k);
    
end

save ff_transfer ff_transfer;

fid_weights=fopen('ff_BW.txt','wt');
fid_bias=fopen('ff_Bb.txt','wt');

for i=1:10
    for j=1:512
        fprintf(fid_weights,'%d',ff_transfer.BW{i}(j));
    end
    fprintf(fid_weights,'\n');
end

for i=1:10
    fprintf(fid_bias,'%d\n',ff_transfer.Bb{i});
end

fclose(fid_weights);
fclose(fid_bias);
    

