function net = cnnbp(net, y)
    n = numel(net.layers);
    
    if strcmp(net.layers{n}.objective, 'sigm')
        %   error
        net.e = net.o - y;
        %  loss function
        net.L = gather(1/2* sum(net.e(:) .^ 2) / size(net.e, 2));

        net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
        net.fvd = (net.ffBW' * net.od);   %feature vector delta
    elseif strcmp(net.layers{n}.objective, 'softmax')
        %   error
        %size(net.layers{1}.a{1},3)
        net.e = -1 * (y - net.o)/size(net.layers{1}.a{1},3);   %��Ϊsoftmaxʱ���������ȳ���batchsize��С
        %  loss function
        net.L = -1 * mean(sum(y.*log(net.o)));   %��������ʧ����ƽ��ֵ��

        %%  backprop deltas
        net.od = net.e;   %  output delta,��������ʧ������ŵ㣬ֱ�ӵõ������������
        net.fvd = (net.ffBW' * net.od) * size(net.layers{1}.a{1},3);              %  feature vector delta
    end;

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n-1}.a{1});
    fvnum = sa(1) * sa(2);
    for i = 1 : numel(net.layers{n-1}.a)
        net.layers{n-1}.d{i} = reshape(net.fvd(((i - 1) * fvnum + 1) : i * fvnum, :), sa(1), sa(2), sa(3));
        m = size(net.layers{n-1}.a{1});
        if strcmp(net.layers{n-1}.type, 's')         %  only conv layers has sigm function
           if strcmp(net.layers{n-1}.activation, 'sign')                        
              %da = ones(size(net.layers{n-1}.a_hat{j}));
              da = (net.layers{n-1}.a_hat{i}>=-1)&(net.layers{n-1}.a_hat{i}<=1);
              net.layers{n-1}.d{i} = net.layers{n-1}.d{i} .* net.layers{n-1}.dropoutMask{i} .* da;
           elseif strcmp(net.layers{n-1}.activation, 'tanh')
              net.layers{n-1}.d{i} = net.layers{n-1}.d{i} .* net.layers{n-1}.dropoutMask{i} .* (net.layers{n-1}.p{i} .* (1 - net.layers{n-1}.p{i}));% need to be exploited
           elseif strcmp(net.layers{n-1}.activation, 'ReLU')
              net.layers{n-1}.d{i} = net.layers{n-1}.d{i} .* net.layers{n-1}.dropoutMask{i} .* (net.layers{n-1}.p{i} > 0);
           end;                           
        end
        %�˴����Լ���BN�ķ��򴫲� 
        if net.useBatchNormalization
           %��0��ֵ1������x~�ĵ���
           d_xnormal = bsxfun(@times, net.layers{n-1}.d{i}, net.layers{n-1}.gamma{i});
           x_mu = bsxfun(@minus, net.layers{n-1}.a_pre{i}, net.layers{n-1}.mu{i});
           inv_sqrt_sigma = 1 ./ sqrt(net.layers{n-1}.sigma2{i} + net.epsilon);
           d_sigma2 =  -0.5 * sum(sum(sum(d_xnormal .* x_mu))) .* inv_sqrt_sigma.^3;%�Է���ĵ���
           d_mu = bsxfun(@times, d_xnormal, inv_sqrt_sigma);
           d_mu = -1 * sum(sum(sum(d_mu))) - 2 * d_sigma2 .* mean(x_mu(:));%�Ծ�ֵ�ĵ���
           net.layers{n-1}.a_norm{i} = net.layers{n-1}.d{i} .* net.layers{n-1}.a_norm{i};%��BN����ĵ���
           %����ط��øĳ�normalize֮���ֵ��ԭ���Ĵ�������
           d_gamma = sum(sum(sum(net.layers{n-1}.a_norm{i})))/m(3);%������Զ��٣��д���                    
           d_beta = sum(sum(sum(net.layers{n-1}.d{i})))/m(3);                      
           di1 = bsxfun(@times,d_xnormal,inv_sqrt_sigma);
           di2 = 2/(m(1)*m(2)*m(3)) * bsxfun(@times, x_mu,d_sigma2);
           net.layers{n-1}.d{i} = di1 + di2 + 1/(m(1)*m(2)*m(3)) * d_mu;
           net.layers{n-1}.dBN{i} = [d_gamma d_beta];              
        end        
    end

    for l = (n - 2) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            mapsize=size(net.layers{l + 1}.d{1});   
            m = size(net.layers{l}.a{1});
            for j = 1 : numel(net.layers{l + 1}.a)
                net.layers{l + 1}.dp{j}=zeros(mapsize(1)*net.layers{l + 1}.scale-net.layers{l + 1}.scale+1,mapsize(2)*net.layers{l + 1}.scale-net.layers{l + 1}.scale+1,mapsize(3));
                net.layers{l + 1}.dp{j}(1:net.layers{l + 1}.scale:end,1:net.layers{l + 1}.scale:end,:)=net.layers{l + 1}.d{j};
            end
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                if strcmp(net.layers{l}.activation, 'sign')
                    %da = ones(size(net.layers{l}.a_hat{j}));
                    da = (net.layers{l}.a_hat{i}>=-1)&(net.layers{l}.a_hat{i}<=1);
                elseif strcmp(net.layers{l}.activation, 'tanh')
                    da = net.layers{l}.a{i} .* (1 - net.layers{l}.a{i});% need to be exploited
                elseif strcmp(net.layers{l}.activation, 'ReLU')
                    da = ( net.layers{l}.a{i} > 0);
                end;                        
                for j = 1 : numel(net.layers{l + 1}.a)
                    z = z + convn(net.layers{l + 1}.dp{j}, rot180(net.layers{l + 1}.Bk{i}{j}), 'full');
                end             
                net.layers{l}.d{i} = da .* z;    
                if net.useBatchNormalization
                   %��0��ֵ1������x~�ĵ���
                   d_xnormal = bsxfun(@times, net.layers{l}.d{i}, net.layers{l}.gamma{i});
                   x_mu = bsxfun(@minus, net.layers{l}.a_pre{i}, net.layers{l}.mu{i});
                   inv_sqrt_sigma = 1 ./ sqrt(net.layers{l}.sigma2{i} + net.epsilon);
                   d_sigma2 =  -0.5 * sum(sum(sum(d_xnormal .* x_mu))) .* inv_sqrt_sigma.^3;%�Է���ĵ���
                   d_mu = bsxfun(@times, d_xnormal, inv_sqrt_sigma);
                   d_mu = -1 * sum(sum(sum(d_mu))) - 2 * d_sigma2 .* mean(x_mu(:));%�Ծ�ֵ�ĵ���
                   net.layers{l}.a_norm{i} = net.layers{l}.d{i} .* net.layers{l}.a_norm{i};%��BN����ĵ���
                   %����ط��øĳ�normalize֮���ֵ��ԭ���Ĵ�������
                   d_gamma = sum(sum(sum(net.layers{l}.a_norm{i})))/m(3);%������Զ��٣��д���                    
                   d_beta = sum(sum(sum(net.layers{l}.d{i})))/m(3);                      
                   di1 = bsxfun(@times,d_xnormal,inv_sqrt_sigma);
                   di2 = 2/(m(1)*m(2)*m(3)) * bsxfun(@times, x_mu,d_sigma2);
                   net.layers{l}.d{i} = di1 + di2 + 1/(m(1)*m(2)*m(3)) * d_mu;
                   net.layers{l}.dBN{i} = [d_gamma d_beta];              
                end                  
            end
            
        elseif strcmp(net.layers{l}.type, 's')
            m = size(net.layers{l}.a{1});
            for i = 1 : numel(net.layers{l}.p)
                z = zeros(size(net.layers{l}.p{1}));
                if strcmp(net.layers{l}.activation, 'sign')
                   %da = ones(size(net.layers{l}.a_hat{i}));
                   da = (net.layers{l}.a_hat{i}>=-1)&(net.layers{l}.a_hat{i}<=1);
                elseif strcmp(net.layers{l}.activation, 'tanh')
                   da = net.layers{l}.p{i} .* (1 - net.layers{l}.p{i});% need to be exploited
                elseif strcmp(net.layers{l}.activation, 'ReLU')
                   da = (net.layers{l}.p{i} > 0);
                end;               
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.Bk{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z .* net.layers{l}.dropoutMask{i} .* da;
                %������Լ���BN�ķ��򴫲�   
                if net.useBatchNormalization
                   %��0��ֵ1������x~�ĵ���
                   d_xnormal = bsxfun(@times, net.layers{l}.d{i}, net.layers{l}.gamma{i});
                   x_mu = bsxfun(@minus, net.layers{l}.a_pre{i}, net.layers{l}.mu{i});
                   inv_sqrt_sigma = 1 ./ sqrt(net.layers{l}.sigma2{i} + net.epsilon);
                   d_sigma2 =  -0.5 * sum(sum(sum(d_xnormal .* x_mu))) .* inv_sqrt_sigma.^3;%�Է���ĵ���
                   d_mu = bsxfun(@times, d_xnormal, inv_sqrt_sigma);
                   d_mu = -1 * sum(sum(sum(d_mu))) - 2 * d_sigma2 .* mean(x_mu(:));%�Ծ�ֵ�ĵ���
                   net.layers{l}.a_norm{i} = net.layers{l}.d{i} .* net.layers{l}.a_norm{i};%��BN����ĵ���
                   %����ط��øĳ�normalize֮���ֵ��ԭ���Ĵ�������
                   d_gamma = sum(sum(sum(net.layers{l}.a_norm{i})))/m(3);%������Զ��٣��д���                    
                   d_beta = sum(sum(sum(net.layers{l}.d{i})))/m(3);                      
                   di1 = bsxfun(@times,d_xnormal,inv_sqrt_sigma);
                   di2 = 2/(m(1)*m(2)*m(3)) * bsxfun(@times, x_mu,d_sigma2);
                   net.layers{l}.d{i} = di1 + di2 + 1/(m(1)*m(2)*m(3)) * d_mu;
                   net.layers{l}.dBN{i} = [d_gamma d_beta];              
                end                 
                
            end
        end
    end

    %%  calc gradients
    for l = 2 : n - 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                %net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.dp{j}, 'valid') / size(net.layers{l}.dp{j}, 3);             
                end    
                %net.layers{l}.db{j} = sum(net.layers{l}.dp{j}(:)) / size(net.layers{l}.dp{j}, 3);%��ʵ��net.layers{l}.d{j}Ҳһ���������������ʵ���㣬��������ı���һ��
            end
        end
    end 
    
    if strcmp(net.layers{n}.objective, 'sigm')
        net.dffW = net.od * (net.fv)' / size(net.od, 2);
        net.dffb = mean(net.od, 2);
    elseif strcmp(net.layers{n}.objective, 'softmax')
        net.dffW = net.od * (net.fv)';
        net.dffb = sum(net.od, 2);
    end;
    

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
