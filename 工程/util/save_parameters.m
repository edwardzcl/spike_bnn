function save_parameters(cnn)
    %feedforward for test
    %clear;
    %clc;
    %load bcnn50BNF
    n = numel(cnn.layers);
    inputmaps = 1;
    for l=2:n-1
        for j=1:cnn.layers{l}.outputmaps      
            for i = 1 : inputmaps
                weights{l}(:,:,i,j)=flipdim(flipdim(cnn.layers{l}.Bk{i}{j}, 1), 2);
            end
            Leakage{l}(j)=cnn.Leakage{l}{j};
        end
        inputmaps=cnn.layers{l}.outputmaps;
    end
    save weights weights;
    save Leakage Leakage;
end