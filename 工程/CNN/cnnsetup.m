function net = cnnsetup(net, x, y)
%     assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));
    global useGpu;

    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's')
            mapsize = mapsize / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            net.layers{l}.inputmaps = inputmaps;
            fan_out = net.layers{l}.outputmaps * net.layers{l}.scale ^ 2;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * net.layers{l}.scale ^ 2;
                for i = 1 : inputmaps
                    if useGpu
                       net.layers{l}.k{i}{j} = (gpuArray.rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                       net.layers{l}.Bk{i}{j} = htanh(net.layers{l}.k{i}{j});
                       net.layers{l}.vk{i}{j} = gpuArray.zeros(net.layers{l}.kernelsize);
                    else
                       net.layers{l}.k{i}{j} = (rand(net.layers{l}.scale) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                       net.layers{l}.Bk{i}{j} = htanh(net.layers{l}.k{i}{j});
                       net.layers{l}.vk{i}{j} = zeros(net.layers{l}.scale);
                       %net.layers{l}.rk{i}{j} = zeros(net.layers{l}.scale);%����Adadelta�㷨�Ŀɱ�ѧϰ��
                    end;
                end
%                 if useGpu
%                     net.layers{l}.b{j} = (gpuArray.rand(1,1) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
%                     net.layers{l}.Bb{j} = htanh(net.layers{l}.b{j});
%                     net.layers{l}.vb{j} = gpuArray.zeros(1,1);
%                 else
%                     net.layers{l}.b{j} = (rand(1,1) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
%                     net.layers{l}.Bb{j} = htanh(net.layers{l}.b{j});
%                     net.layers{l}.vb{j} = 0;
%                     %net.layers{l}.rb{j} = 0;
%                 end;
                %�˴����Լ���BN����ĳ�ʼ��
                if net.useBatchNormalization
                   net.layers{l}.mu{j} = 0;
                   net.layers{l}.sigma2{j} = 1;
                   net.layers{l}.gamma{j} = 1;
                   net.layers{l}.beta{j} = 0;
                   net.layers{l}.vBN{j} = zeros(1,2);
                   %net.layers{l}.rBN{j} = zeros(1,2);
                   net.layers{l}.mean_sigma2{j} = 0;
                   net.layers{l}.mean_mu{j} = 0; 
                end                         
            end
            inputmaps = net.layers{l}.outputmaps;
        end
        
        if strcmp(net.layers{l}.type, 'c')
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            net.layers{l}.inputmaps = inputmaps;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map
                    if useGpu
                        net.layers{l}.k{i}{j} = (gpuArray.rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                        net.layers{l}.Bk{i}{j}=htanh(net.layers{l}.k{i}{j});
                        net.layers{l}.vk{i}{j} = gpuArray.zeros(net.layers{l}.kernelsize);
                    else
                        net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                        net.layers{l}.Bk{i}{j}=htanh(net.layers{l}.k{i}{j});
                        net.layers{l}.vk{i}{j} = zeros(net.layers{l}.kernelsize);%һ�ε��ĸ�����
                        %net.layers{l}.rk{i}{j} = zeros(net.layers{l}.kernelsize);%����Adadelta�㷨�Ŀɱ�ѧϰ��
                    end;
                end
%                 if useGpu
%                     net.layers{l}.b{j} = (gpuArray.rand(1,1) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
%                     net.layers{l}.Bb{j} = htanh(net.layers{l}.b{j});
%                     net.layers{l}.vb{j} = gpuArray.zeros(1,1);
%                 else
%                     net.layers{l}.b{j} = (rand(1,1) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
%                     net.layers{l}.Bb{j}=htanh(net.layers{l}.b{j});
%                     net.layers{l}.vb{j} = 0;
%                     %net.layers{l}.rb{j} = 0;    
%                 end;
                if net.useBatchNormalization
                   net.layers{l}.mu{j} = 0;
                   net.layers{l}.sigma2{j} = 1;
                   net.layers{l}.gamma{j} = 1;
                   net.layers{l}.beta{j} = 0;
                   net.layers{l}.vBN{j} = zeros(1,2);
                   %net.layers{l}.rBN{j} = zeros(1,2);
                   net.layers{l}.mean_sigma2{j} = 0;
                   net.layers{l}.mean_mu{j} = 0; 
                end
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);
    if useGpu
        net.ffb = gpuArray.zeros(onum, 1);
        net.ffBb=htanh(net.ffb);%�����ʼ��ƫ�ƣ����ܻᵼ�¹��
        net.vffb = gpuArray.zeros(onum, 1);
        net.ffW = (gpuArray.rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
        net.ffBW=htanh(net.ffW);
        net.vffW = gpuArray.zeros(onum, fvnum);
    else
        net.ffb = zeros(onum, 1);
        net.ffBb=htanh(net.ffb);%�����ʼ��ƫ�ƣ����ܻᵼ�¹��
        net.vffb = zeros(onum, 1);
        %net.rffb = zeros(onum, 1);
        net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
        net.ffBW=htanh(net.ffW);
        net.vffW = zeros(onum, fvnum);
        %net.rffW = zeros(onum, fvnum);
    end;
end
