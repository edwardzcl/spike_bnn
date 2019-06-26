load mnist_uint8;

%归一化
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y');
test_y = double(test_y');

% normalize，0中心化，1方差化
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);
train_x(find(train_x>0))=1;
train_x(find(train_x<=0))=0;
test_x(find(test_x>0))=1;
test_x(find(test_x<=0))=0;

train_x = reshape(train_x',28,28,60000);
test_x = reshape(test_x',28,28,10000);

%train_data
for k=1:60000
    for j=1:3
        for i=1:3
            train_data{k}{(j-1)*3+i}=train_x((j-1)*8+1:(j-1)*8+1+11,(i-1)*8+1:(i-1)*8+1+11,k);
        end
    end
end

%test_data
for k=1:10000
    for j=1:3
        for i=1:3
            test_data{k}{(j-1)*3+i}=test_x((j-1)*8+1:(j-1)*8+1+11,(i-1)*8+1:(i-1)*8+1+11,k);
        end
    end
end
