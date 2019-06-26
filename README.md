# spike_bnn
### matlab版本二值神经网络，可转换直接用于脉冲神经网络


要点：
1、把输入已经转置(已经不可辨哈哈哈)<br>
2、权值泄露没问题，已经处理过<br>
3、脉冲，膜电位也是按照转置过脉冲网络正向生成的<br>
4、自定义1800为500+1300<br>
5、mnist数据集已经转置，并且0/1二值化；503字符集bmp已经是0/1二值化，但需要把1变成0，0变成1；1300自定义按照>=255为0，<255为1<br>
6、mnist test有分块，自定义1800没有分块(训练测试相同)，参见input_data.m<br>
7、retrain之后的error_rate，mnist：  84.91%；  1800自定义：98.83%   错误标号参见bad_for_mnist，bad_for_1800<br>
8、ff_parameter也是处理好的，包括两个txt文件，参见core_input_parameter.m<br>
