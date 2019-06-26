function net = cnntrain(net, x, y, test_x, test_y, opts)
    global useGpu;
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
%         tic;
        if net.useBatchNormalization
           for k = 1 : numel(net.layers)
               if strcmp(net.layers{k}.type, 'c')||strcmp(net.layers{k}.type, 's')
                  for j = 1 : net.layers{k}.outputmaps
                      net.layers{k}.mean_sigma2{j} = 0;
                      net.layers{k}.mean_mu{j} = 0;
                  end
               end
           end;
        end
        
        kk = randperm(m);
        for l = 1 : numbatches
            if useGpu
                batch_x = gpuArray(x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)));
                batch_y = gpuArray(y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)));
            else
                batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
                batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            end;
            
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = gather(net.L);
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * gather(net.L); 
            
            if net.useBatchNormalization
               for k = 1 : numel(net.layers)
                   if strcmp(net.layers{k}.type, 'c')||strcmp(net.layers{k}.type, 's')
                      for j = 1 : net.layers{k}.outputmaps
                          net.layers{k}.mean_sigma2{j} = net.layers{k}.mean_sigma2{j} + net.layers{k}.sigma2{j};
                          net.layers{k}.mean_mu{j} = net.layers{k}.mean_mu{j} + net.layers{k}.mu{j};
                      end
                   end
               end;  
            end
            
            if mod(l,10)==0
                disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) 'batch ' num2str(l) '/' num2str(numbatches)]);
            end;
        end
        
        if net.useBatchNormalization
           for k = 1 : numel(net.layers)
               if strcmp(net.layers{k}.type, 'c')||strcmp(net.layers{k}.type, 's')
                  for j = 1 : net.layers{k}.outputmaps
                      net.layers{k}.mean_sigma2{j} = net.layers{k}.mean_sigma2{j} / (numbatches - 1);
                      net.layers{k}.mean_mu{j} = net.layers{k}.mean_mu{j} / numbatches;
                  end
               end
           end;   
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ' error:' num2str(net.L)]);
%         toc;
        [er, bad] = cnntest(net, test_x, test_y);
        if i==1
           net.error_rate(i)=er;
        else
           if er<=net.error_rate(end)
              net.error_rate(end+1)=er;
              save best_cnn net ; 
           end
        end
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) 'error_rate:' num2str(er)]);
    end
    
end
