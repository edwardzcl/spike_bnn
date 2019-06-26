function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;
    global useGpu;

    for l = 2 : n-1   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                if useGpu
                    z = gpuArray.zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                else
                    z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                end;
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.Bk{i}{j}, 'valid');
                end;
                %  add bias, pass through nonlinearity
                net.layers{l}.a_pre{j} = z ;% + net.layers{l}.Bb{j};%ԭ��˼ӵõ���x
                %  ����BN����
                if net.useBatchNormalization
                   if net.testing
                      norm_factor = net.layers{l}.gamma{j}./sqrt(net.layers{l}.mean_sigma2{j}+net.epsilon);
                      net.layers{l}.a_hat{j} = bsxfun(@times, net.layers{l}.a_pre{j}, norm_factor);
                      net.layers{l}.a_hat{j} = bsxfun(@plus, net.layers{l}.a_hat{j}, net.layers{l}.beta{j} -  norm_factor .* net.layers{l}.mean_mu{j});
                   else
                      net.layers{l}.mu{j} = mean(net.layers{l}.a_pre{j}(:));
                      x_mu = bsxfun(@minus,net.layers{l}.a_pre{j},net.layers{l}.mu{j}).^2;
                      net.layers{l}.sigma2{j} = mean(x_mu(:));
                      net.layers{l}.a_norm{j} = bsxfun(@minus,net.layers{l}.a_pre{j},net.layers{l}.mu{j});
                      net.layers{l}.a_norm{j} = bsxfun(@rdivide, net.layers{l}.a_norm{j}, sqrt(net.layers{l}.sigma2{j}+net.epsilon));
                      net.layers{l}.a_hat{j} = bsxfun(@plus, net.layers{l}.a_norm{j} .* net.layers{l}.gamma{j}, net.layers{l}.beta{j});
                   end;
                else
                    net.layers{l}.a_hat{j} = net.layers{l}.a_pre{j};
                end                                   
                if strcmp(net.layers{l}.activation, 'sign')
                    net.layers{l}.a{j} = step(net.layers{l}.a_hat{j});
                elseif strcmp(net.layers{l}.activation, 'tanh')
                    net.layers{l}.a{j} = tanh(z + net.layers{l}.Bb{j});% need to be exploited
                elseif strcmp(net.layers{l}.activation, 'ReLU')
                    net.layers{l}.a{j} = max(z + net.layers{l}.Bb{j},0);
                end;
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
               %  downsample
               %  !!below can probably be handled by insane matrix operations
               for j = 1 : net.layers{l}.outputmaps   %  for each output map
                   %  create temp output map
                   if useGpu
                      %z = gpuArray.zeros(size(net.layers{l - 1}.a{1})./[net.layers{l}.scale net.layers{l}.scale 1]);
                      z = gpuArray.zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.scale - 1 net.layers{l}.scale - 1 0]);
                   else
                      %z = zeros(size(net.layers{l - 1}.a{1})./[net.layers{l}.scale net.layers{l}.scale 1]);
                      z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.scale - 1 net.layers{l}.scale - 1 0]);
                   end;
                   for i = 1 : inputmaps   %  for each input map
                       %  convolve with corresponding kernel and add to temp output map
                       z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.Bk{i}{j}, 'valid');
                   end;               
                   net.layers{l}.a_pre{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :) ;%+net.layers{l}.Bb{j};
                   %  �˴���������BN
                   if net.useBatchNormalization
                      if net.testing
                         norm_factor = net.layers{l}.gamma{j}./sqrt(net.layers{l}.mean_sigma2{j}+net.epsilon);
                         net.layers{l}.a_hat{j} = bsxfun(@times, net.layers{l}.a_pre{j}, norm_factor);
                         net.layers{l}.a_hat{j} = bsxfun(@plus, net.layers{l}.a_hat{j}, net.layers{l}.beta{j} -  norm_factor .* net.layers{l}.mean_mu{j});
                      else
                         net.layers{l}.mu{j} = mean(net.layers{l}.a_pre{j}(:));
                         x_mu = bsxfun(@minus,net.layers{l}.a_pre{j},net.layers{l}.mu{j}).^2;
                         net.layers{l}.sigma2{j} = mean(x_mu(:));
                         net.layers{l}.a_norm{j} = bsxfun(@minus,net.layers{l}.a_pre{j},net.layers{l}.mu{j});
                         net.layers{l}.a_norm{j} = bsxfun(@rdivide, net.layers{l}.a_norm{j}, sqrt(net.layers{l}.sigma2{j}+net.epsilon));
                         net.layers{l}.a_hat{j} = bsxfun(@plus, net.layers{l}.a_norm{j} .* net.layers{l}.gamma{j}, net.layers{l}.beta{j});
                      end;
                   else                                                                  
                      net.layers{l}.a_hat{j} = net.layers{l}.a_pre{j};  
                   end
                   %  add bias, pass through nonlinearity
                   if strcmp(net.layers{l}.activation, 'sign')
                      net.layers{l}.p{j} = step(net.layers{l}.a_hat{j});
                   elseif strcmp(net.layers{l}.activation, 'tanh')
                      net.layers{l}.p{j} = tanh(z + net.layers{l}.Bb{j});% need to be exploited
                   elseif strcmp(net.layers{l}.activation, 'ReLU')
                      net.layers{l}.p{j} = max(z + net.layers{l}.Bb{j},0);
                   end;
                   net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.p{j}))>net.dropoutFraction);
                   net.layers{l}.a{j} = net.layers{l}.p{j} .* net.layers{l}.dropoutMask{j};
               end;
               inputmaps = net.layers{l}.outputmaps;        

                    
        end;          
    end;

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n-1}.a)
        sa = size(net.layers{n-1}.a{j});
        net.fv = [net.fv; reshape(net.layers{n-1}.a{j}, sa(1) * sa(2), sa(3))];%�Ѷ�ά����ͼչ��һ������
    end;
    %  feedforward into output perceptrons
    if strcmp(net.layers{n}.objective, 'sigm')
        net.o = sigm(net.ffBW * net.fv + repmat(net.ffBb, 1, size(net.fv, 2)));
    elseif strcmp(net.layers{n}.objective, 'softmax')
        M = net.ffBW*net.fv;
        M = bsxfun(@plus, M, net.ffBb);
        M = bsxfun(@minus, M, max(M, [], 1));
        M = exp(M);
        M = bsxfun(@rdivide, M, sum(M));
        net.o = M;
    end;

end
