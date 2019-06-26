% function test_example_CNN
clear;
%load mnist_uint8
load bcnn50BNF1(����)
load digital_custom
global useGpu;%ȫ���źţ�һ���������ദ����
useGpu = false;%����GPU
%**************************************************************
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y');
% test_y = double(test_y');
% 
% % normalize
% [train_x, mu, sigma] = zscore(train_x);
% test_x = normalize(test_x, mu, sigma);
% train_x(find(train_x>0))=1;
% train_x(find(train_x<=0))=0;
% test_x(find(test_x>0))=1;
% test_x(find(test_x<=0))=0;
% 
% 
% train_x = reshape(train_x',28,28,60000);
% test_x = reshape(test_x',28,28,10000);

%*********************************************************************%

train_x = permute(digital_custom.train_x,[2,1,3]);
train_y=digital_custom.train_y;
test_x=train_x;
test_y=digital_custom.train_y;


opts.alpha = 0.01;%ȫ��ѧϰ��
opts.alphascale = 0.5;%ѧϰ��˥����,���������ﲢû������
opts.batchsize = 50;
opts.numepochs = 50;
opts.momentum = 0.5;%������
opts.momIncrease = 12000;%����������ʹ�õ�����
cnn.iter = 0;
cnn.error_rate=[];
cnn.testing = false;
cnn.useBatchNormalization = 1;
cnn.epsilon = 1e-10;%  numeric factor for batch normalization
cnn.dropoutFraction = 0;%��ʹ��dropout

%cnn = cnnsetup(cnn, train_x, train_y);
% cnn = cnnff(cnn,train_x(:,:,1:1000));
cnn = cnntrain(cnn, train_x, train_y, test_x, test_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
