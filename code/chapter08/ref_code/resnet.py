import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential

# ResNet基本块定义
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # 形状的的转换
        # stride > 1,相当于下采样
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x
    # 前向传播函数，[b, h, w, c]
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        identity = self.downsample(inputs)
        # f(x)+x的运算
        add = layers.add([out, identity])
        # 不建议使用 self.relu()
        output = tf.nn.relu(add)
        return output
# ResNet18 类的创建   
class ResNet(keras.Model):
    # [2,2,2,2]
    def __init__(self, layer_dims,  num_classes=10):
        super(ResNet, self).__init__()
        # 第一层
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)), 
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 64, 128等表示通道数
        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        
        # [b ,512, h, w]=>[b, 512, 1, 1]
        # 输出层
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)
        
    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)

        return x
    
    def build_resblock(self, filter_num, blocks, stride=1):
        # blocks残差层块数,如blocks=2
        res_block = Sequential()
        res_block.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, stride=1))
        return res_block       

def resnet18():
    # layer_dims
    # 1+ 2×2+2×2+2×2+2×2 +1 =18
    return ResNet([2, 2, 2, 2])

def resnet34():
    return ResNet([3, 4, 6, 3])