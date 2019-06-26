function net = cnnapplygrads(net, opts)
    mom = 0.5;
    net.iter = net.iter +1;
    if net.iter > opts.momIncrease
        mom = opts.momentum;
    end;
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    %net.layers{l}.rk{ii}{j} = 0.9 * net.layers{l}.rk{ii}{j} + 0.1 * net.layers{l}.dk{ii}{j}.^2;
                    % net.layers{l}.dk{ii}{j} = opts.alpha * net.layers{l}.dk{ii}{j} ./ (sqrt(net.layers{l}.rk{ii}{j}+net.epsilon));
                    %net.layers{l}.vk{ii}{j} = mom * net.layers{l}.vk{ii}{j} + net.layers{l}.dk{ii}{j};
                    
                    net.layers{l}.vk{ii}{j} = mom * net.layers{l}.vk{ii}{j} + opts.alpha * net.layers{l}.dk{ii}{j};
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - net.layers{l}.vk{ii}{j};
                    net.layers{l}.k{ii}{j}(find(net.layers{l}.k{ii}{j}>=1))=1;
                    net.layers{l}.k{ii}{j}(find(net.layers{l}.k{ii}{j}<=-1))=-1;
                    net.layers{l}.Bk{ii}{j}=htanh(net.layers{l}.k{ii}{j});
                end
                %net.layers{l}.rb{j} = 0.9 * net.layers{l}.rb{j} + 0.1 * net.layers{l}.db{j}.^2;
                %net.layers{l}.db{j} = opts.alpha * net.layers{l}.db{j} ./ (sqrt(net.layers{l}.rb{j}+net.epsilon));             
                %net.layers{l}.vb{j} = mom * net.layers{l}.vb{j} + net.layers{l}.db{j};
                
%                 net.layers{l}.vb{j} = mom * net.layers{l}.vb{j} + opts.alpha * net.layers{l}.db{j};               
%                 net.layers{l}.b{j} = net.layers{l}.b{j} - net.layers{l}.vb{j};
%                 net.layers{l}.b{j}(find(net.layers{l}.b{j}>=1))=1;
%                 net.layers{l}.b{j}(find(net.layers{l}.b{j}<=-1))=-1;
%                 net.layers{l}.Bb{j}=htanh(net.layers{l}.b{j});
                
                if net.useBatchNormalization               
                   %net.layers{l}.rBN{j} = 0.9 * net.layers{l}.rBN{j} + 0.1 * net.layers{l}.dBN{j}.^2;
                   %net.layers{l}.dBN{j} = opts.alpha * net.layers{l}.dBN{j} ./ (sqrt(net.layers{l}.rBN{j}+net.epsilon));            
                   %net.layers{l}.vBN{j} = mom * net.layers{l}.vBN{j} + net.layers{l}.dBN{j};
                  
                   net.layers{l}.vBN{j} = mom * net.layers{l}.vBN{j} + opts.alpha * net.layers{l}.dBN{j};
                   net.layers{l}.gamma{j} = net.layers{l}.gamma{j} - net.layers{l}.vBN{j}(1:length(net.layers{l}.gamma{j}));
                   net.layers{l}.beta{j} = net.layers{l}.beta{j} - net.layers{l}.vBN{j}(length(net.layers{l}.gamma{j})+1:end);
                end;                 
            end
        elseif strcmp(net.layers{l}.type, 's')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    %net.layers{l}.rk{ii}{j} = 0.9 * net.layers{l}.rk{ii}{j} + 0.1 * net.layers{l}.dk{ii}{j}.^2;
                    % net.layers{l}.dk{ii}{j} = opts.alpha * net.layers{l}.dk{ii}{j} ./ (sqrt(net.layers{l}.rk{ii}{j}+net.epsilon));
                    %net.layers{l}.vk{ii}{j} = mom * net.layers{l}.vk{ii}{j} + net.layers{l}.dk{ii}{j};
                    
                    net.layers{l}.vk{ii}{j} = mom * net.layers{l}.vk{ii}{j} + opts.alpha * net.layers{l}.dk{ii}{j};
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - net.layers{l}.vk{ii}{j};
                    net.layers{l}.k{ii}{j}(find(net.layers{l}.k{ii}{j}>=1))=1;
                    net.layers{l}.k{ii}{j}(find(net.layers{l}.k{ii}{j}<=-1))=-1;
                    net.layers{l}.Bk{ii}{j}=htanh(net.layers{l}.k{ii}{j});
                end
                %net.layers{l}.rb{j} = 0.9 * net.layers{l}.rb{j} + 0.1 * net.layers{l}.db{j}.^2;
                %net.layers{l}.db{j} = opts.alpha * net.layers{l}.db{j} ./ (sqrt(net.layers{l}.rb{j}+net.epsilon));             
                %net.layers{l}.vb{j} = mom * net.layers{l}.vb{j} + net.layers{l}.db{j};
                
%                 net.layers{l}.vb{j} = mom * net.layers{l}.vb{j} + opts.alpha * net.layers{l}.db{j};               
%                 net.layers{l}.b{j} = net.layers{l}.b{j} - net.layers{l}.vb{j};
%                 net.layers{l}.b{j}(find(net.layers{l}.b{j}>=1))=1;
%                 net.layers{l}.b{j}(find(net.layers{l}.b{j}<=-1))=-1;
%                 net.layers{l}.Bb{j}=htanh(net.layers{l}.b{j});
                %�˴����Լ���BN��ѧϰ����
                if net.useBatchNormalization               
                   %net.layers{l}.rBN{j} = 0.9 * net.layers{l}.rBN{j} + 0.1 * net.layers{l}.dBN{j}.^2;
                   %net.layers{l}.dBN{j} = opts.alpha * net.layers{l}.dBN{j} ./ (sqrt(net.layers{l}.rBN{j}+net.epsilon));            
                   %net.layers{l}.vBN{j} = mom * net.layers{l}.vBN{j} + net.layers{l}.dBN{j};
                  
                   net.layers{l}.vBN{j} = mom * net.layers{l}.vBN{j} + opts.alpha * net.layers{l}.dBN{j};
                   net.layers{l}.gamma{j} = net.layers{l}.gamma{j} - net.layers{l}.vBN{j}(1:length(net.layers{l}.gamma{j}));
                   net.layers{l}.beta{j} = net.layers{l}.beta{j} - net.layers{l}.vBN{j}(length(net.layers{l}.gamma{j})+1:end);
                end;  
            end 
        end
    end            
    
    %net.rffW = 0.9 * net.rffW + 0.1 * net.dffW.^2;
    %net.dffW = opts.alpha * net.dffW ./ (sqrt(net.rffW+net.epsilon));    
    %net.vffW = mom * net.vffW + net.dffW;
   
    net.vffW = mom * net.vffW + opts.alpha * net.dffW;
    net.ffW = net.ffW - net.vffW;
    net.ffW(find(net.ffW>=1))=1;
    net.ffW(find(net.ffW<=-1))=-1;
    net.ffBW=htanh(net.ffW);
    
    %net.rffb = 0.9 * net.rffb + 0.1 * net.dffb.^2;
    %net.dffb = opts.alpha * net.dffb ./ (sqrt(net.rffb+net.epsilon)); 
    %net.vffb = mom * net.vffb + net.dffb;
   
    net.vffb = mom * net.vffb + opts.alpha * net.dffb;
    net.ffb = net.ffb - net.vffb;
    net.ffb(find(net.ffb>=1))=1;
    net.ffb(find(net.ffb<=-1))=-1;
    net.ffBb=htanh(net.ffb);
end
