function cnn=bn_int(cnn)
    for i=2:numel(cnn.layers)-1
        for j=1:cnn.layers{i,1}.outputmaps
%             cnn.layers{i,1}.mean_sigma2{j}=ceil(cnn.layers{i,1}.mean_sigma2{j});
%             cnn.layers{i,1}.mean_mu{j}=floor(cnn.layers{i,1}.mean_mu{j});
%             cnn.layers{i,1}.gamma{j}=round(cnn.layers{i,1}.gamma{j});
%             cnn.layers{i,1}.beta{j}=round(cnn.layers{i,1}.beta{j});
               %如果确定要浮点数，然后最后统�?��整的话，可略去上面的步骤，如果想采用整数做处理，可以保留上面的步骤，然后继续进行处理
            cnn.Leakage{i}{j}=round(cnn.layers{i,1}.beta{j} .* sqrt(cnn.layers{i,1}.mean_sigma2{j}+cnn.epsilon) ./ cnn.layers{i,1}.gamma{j} - cnn.layers{i,1}.mean_mu{j});
        end
    end

%   for i=2:numel(cnn.layers)-1
%       for j=1:cnn.layers{i,1}.outputmaps
%           cnn.layers{i,1}.mean_sigma2{j}=ceil(cnn.layers{i,1}.mean_sigma2{j});
%           cnn.layers{i,1}.mean_mu{j}=floor(cnn.layers{i,1}.mean_mu{j});
%           cnn.layers{i,1}.gamma{j}(find(cnn.layers{i,1}.gamma{j}>0))=1;
%           cnn.layers{i,1}.gamma{j}(find(cnn.layers{i,1}.gamma{j}<0))=-1;
%           cnn.layers{i,1}.beta{j}=round(cnn.layers{i,1}.beta{j});     
%           L{i}{j}=cnn.layers{i,1}.beta{j} .* sqrt(cnn.layers{i,1}.mean_sigma2{j}) ./ cnn.layers{i,1}.gamma{j} - cnn.layers{i,1}.mean_mu{j}; % =round(cnn.layers{i,1}.gamma{j})
%       end
%   end

end