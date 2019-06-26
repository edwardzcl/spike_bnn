clear;
clc;
load ff_parameter;

%flat=zeros(2*2*32*4,1);
final_map=zeros(4,4,32);

for i=1:4
    filename=['output',num2str(i-1),'.txt'];
    fid{i}=fopen(filename,'r');
    core_map{i}=zeros(2,2,32);
end

for i=1:numel(fid)
    line = 1;
    while feof(fid{i}) == 0   
        str_line{i,line} = fgetl(fid{i});
        data_line{i}(line) = bin2dec(str_line{i,line}(27:36))-(i-1)*128+1;
        line = line+1;
    end
    core_map{i}([data_line{i}])=1;
    core_map{i}=permute(core_map{i},[2,1,3]);
end

row1=cat(2,core_map{1},core_map{2});
row2=cat(2,core_map{3},core_map{4});
final_map=cat(1,row1,row2);
M = ff_parameter.ffBW*final_map(:);
M = bsxfun(@plus, M, ff_parameter.ffBb);
[~, h] = max(M);
output_digit=h

for i=1:4
    fclose(fid{i});
end

% [~, a] = max(test_y);
% bad = find(h ~= a);
% er = numel(bad) / size(y, 2);

% for i=1:numel(fid)
%     for j=1:size(data_line{i},2)
%         channel=ceil(data_line{i}(j)/4);
%         spatial=mod(data_line{i}(j),4);
%         
%         
%     end
% end