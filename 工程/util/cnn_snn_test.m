    %  feedforward for test
    %clear;
    %clc;
    %load bcnn50BNF
    
    net=bn_int(net);
    net.testing =true;
    net = cnn_snn_ff(net, test_x);
    [~, h] = max(net.o);
    [~, a] = max(test_y);
    bad = find(h ~= a);

    er = numel(bad) / size(test_y, 2)
    
    save_parameters(net);
