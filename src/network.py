from layer import *
import tensorflow as tf

class Network():
    def __init__(self,x,mask,local_x,global_completion,local_completion,is_training,batch_size):
        #global_completion 全局补全
        #local_completion 局部补全
        self.batch_size=batch_size
        self.imitation=self.generator(x*(1-mask),is_training)#填好洞的结果 mask=1,需要填补 mask=0不需要填补
        self.completion=self.imitation * mask + x * (1-mask) #裁剪部分+非裁剪部分
        self.real=self.discriminator(x,local_x,reuse=False) ##判别网络 实际的图像+局部的图像
        self.fake=self.discriminator(global_completion,local_completion,reuse=True) #reuse=True 再次调用 该共享变量作用域
        self.g_loss=self.calc_g_loss(x,self.completion)
        self.d_loss=self.calc_d_loss(self.real,self.fake)
        #get_collection 从一个集合中取出全部变量
        #tf.GraphKeys.TRAINABLE_VARIABLES 科学系的变量
        self.g_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
        self.d_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')





    def generator(self,x,is_training):
        with tf.variable_scope('generator'):
            #卷机层 6 层
            with tf.variable_scope('conv1'):
                x=conv_layer(x,[5,5,3,64],1)
                x=batch_normalize(x,is_training) #batch_normalize 在神经网络的训练过程中对每层的输入数据加一个标准化处理
                x=tf.nn.relu(x)
            with tf.variable_scope('conv2'):
                x=conv_layer(x,[3,3,64,128],2)
                x=batch_normalize(x,is_training)
                x=tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x=conv_layer(x,[3,3,128,128],1)
                x=batch_normalize(x,is_training)
                x=tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x=conv_layer(x,[3,3,128,256],2)
                x=batch_normalize(x,is_training)
                x=tf.nn.relu(x)
            with tf.variable_scope('conv5'):
                x=conv_layer(x,[3,3,256,256],1)
                x=batch_normalize(x,is_training)
                x=tf.nn.relu(x)
            with tf.variable_scope('conv6'):
                x=conv_layer(x,[3,3,256,256],1)
                x=batch_normalize(x,is_training)
                x=tf.nn.relu(x)

            #空洞卷积 4层
            with tf.variable_scope('dilated1'):
                x=dilated_conv_layer(x,[3,3,256,256],2) #2 2-1=1 添加一个空洞的个数
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated2'):
                x=dilated_conv_layer(x,[3,3,256,256],4)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated3'):
                x=dilated_conv_layer(x,[3,3,256,256],8)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated4'):
                x=dilated_conv_layer(x,[3,3,256,256],16)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)

            #卷机层 2层
            with tf.variable_scope('conv7'):
                x=conv_layer(x,[3,3,256,256],1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv8'):
                x=conv_layer(x,[3,3,256,256],1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)

            #反卷机 1 层
            with tf.variable_scope('deconv1'):
                x=deconv_layer(x,[4,4,128,256],[self.batch_size,64,64,128],2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)

            #卷积 1 层
            with tf.variable_scope('conv9'):
                x=conv_layer(x,[3,3,128,128],1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)

            #反卷机 1 层
            with tf.variable_scope('deconv2'):
                x=deconv_layer(x,[4,4,64,128],[self.batch_size,128,128,64],2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)

            #卷积 2 层
            with tf.variable_scope('conv10'):
                x=conv_layer(x,[3,3,64,32],1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv11'):
                x=conv_layer(x,[3,3,32,3],1)
                x = batch_normalize(x, is_training)
                x = tf.nn.tanh(x)
        return x

    def discriminator(self,global_x,local_x,reuse):

        #global 判别器
        def global_discriminator(x):
            is_training=tf.constant(True)
            with tf.variable_scope('global'):

                #卷机层 5 层
                with tf.variable_scope('conv1'):
                    x=conv_layer(x,[5,5,3,64],2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x=conv_layer(x,[5,5,64,128],2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)

                with tf.variable_scope('conv3'):
                    x=conv_layer(x,[5,5,128,256],2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)

                with tf.variable_scope('conv4'):
                    x=conv_layer(x,[5,5,256,512],2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)

                with tf.variable_scope('conv5'):
                    x=conv_layer(x,[5,5,512,512],2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)

                #全链接层 1 层
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x,1024)
            return x

        #local判别函数
        def local_discriminator(x):
            is_training=tf.constant(True)
            with tf.variable_scope('local'):

                #卷机层 4 层
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)

                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)

                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)

                #全链接层 1 层
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        #global判别函数和local判别函数相连接且通过1层全链接层输出
        with tf.variable_scope('discriminator',reuse=reuse):
            global_output=global_discriminator(global_x)
            local_output=local_discriminator(local_x)
            with tf.variable_scope('concatenation'):
                output=tf.concat((global_output,local_output),1)
                output=full_connection_layer(output,1)
        return output

    #generator 损失函数 MSE loss
    def calc_g_loss(self,x,completion):
        loss=tf.nn.l2_loss(x-completion)
        return tf.reduce_mean(loss)

    #discriminator 损失函数
    def calc_d_loss(self,real,fake):
        alpha=4e-4
        d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real,labels=tf.ones_like(real)))
        #ltf.nn.sigmoid_cross_entropy_with_logits(logits=real,labels=tf.ones_like(real)) 输出为Batch中每个样本的loss

        d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real,d_loss_fake)*alpha

















