%layer1
for layer=2:5
    for i=1:10000
        for j=1:numel(net.layers{layer,1}.a)
            sample_spike{i}.layers{layer}(:,:,j)=net.layers{layer,1}.a{1,j}(:,:,i);
            sample_voltage{i}.layers{layer}(:,:,j)=potential{1,layer}{1,j}(:,:,i);
        end
    end
end

%layer2
for k=1:10000
    for i=1:3
        for j=1:3
            spike{k}.layers{2}.potential{i,j}=sample_spike{k}.layers{2}((i-1)*8+1:(i-1)*8+8,(j-1)*8+1:(j-1)*8+8,:);
            voltage{k}.layers{2}.potential{i,j}=sample_voltage{k}.layers{2}((i-1)*8+1:(i-1)*8+8,(j-1)*8+1:(j-1)*8+8,:);
        end
    end
end

%layer3
for k=1:10000
    for i=1:3
        for j=1:3
            spike{k}.layers{3}.potential{i,j}=sample_spike{k}.layers{3}((i-1)*4+1:(i-1)*4+4,(j-1)*4+1:(j-1)*4+4,:);
            voltage{k}.layers{3}.potential{i,j}=sample_voltage{k}.layers{3}((i-1)*4+1:(i-1)*4+4,(j-1)*4+1:(j-1)*4+4,:);
        end
    end
end
        
%layer4
for k=1:10000
    for i=1:2
        for j=1:2
            spike{k}.layers{4}.potential{i,j}=sample_spike{k}.layers{4}((i-1)*4+1:(i-1)*4+4,(j-1)*4+1:(j-1)*4+4,:);
            voltage{k}.layers{4}.potential{i,j}=sample_voltage{k}.layers{4}((i-1)*4+1:(i-1)*4+4,(j-1)*4+1:(j-1)*4+4,:);
        end
    end
end


%layer5
for k=1:10000
    for i=1:2
        for j=1:2
            spike{k}.layers{5}.potential{i,j}=sample_spike{k}.layers{5}((i-1)*2+1:(i-1)*2+2,(j-1)*2+1:(j-1)*2+2,:);
            voltage{k}.layers{5}.potential{i,j}=sample_voltage{k}.layers{5}((i-1)*2+1:(i-1)*2+2,(j-1)*2+1:(j-1)*2+2,:);
        end
    end
end


