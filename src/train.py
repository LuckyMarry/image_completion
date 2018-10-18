import numpy as np
import tensorflow as tf
import cv2
import tqdm #可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)

from network import Network
import load


# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

IMAGE_SIZE=128 #当前图像的大小
LOCAL_SIZE=64 #对抗生成网络size
HOLE_MIN=24
HOLE_MAX=48
LEARNING_RATE=1e-3
BATCH_SIZE=16
PRETRAIN_EPOCH=20

#epoch：迭代次数，1个epoch等于使用训练集中的全部样本训练一次
#batchsize：中文翻译为批大小（批尺寸）。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
#iteration：中文翻译为迭代，1个iteration等于使用batchsize个样本训练一次；
#一个迭代 = 一个正向通过+一个反向通过
#一个epoch = 所有训练样本的一个正向传递和一个反向传递

def train():
    x=tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])
    mask=tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1])#mask=1 当前位置需要被填充 mask=0 当前位置不需要填充
    local_x=tf.placeholder(tf.float32,[BATCH_SIZE,LOCAL_SIZE,LOCAL_SIZE,3]) #local 局部
    global_completion=tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])
    local_completion=tf.placeholder(tf.float32,[BATCH_SIZE,LOCAL_SIZE,LOCAL_SIZE,3])
    is_training=tf.placeholder(tf.bool,[])
    print (is_training)

    model=Network(x,mask,local_x,global_completion,local_completion,is_training,batch_size=BATCH_SIZE)
    sess=tf.Session()

    global_step=tf.Variable(0,name='global_step',trainable=False)
    epoch=tf.Variable(0,name='epoch',trainable=False)

    opt=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) #创建adam优化器
    g_train_op=opt.minimize(model.g_loss,global_step=global_step,var_list=model.g_variables)
    d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

    init_op=tf.global_variables_initializer() #初始化模型的参数
    sess.run(init_op)

    if tf.train.get_checkpoint_state('./backup'): #通过checkpoint文件找到模型文件名
        saver=tf.train.Saver()
        saver.restore(sess,'./backup/latest')

    x_train,x_test=load.load()
    print (x_train.shape)
    x_train=np.array([a/255 for a in x_train])
    print (x_train[0])
    x_test=np.array([a/255 for a in x_test])

    step_num=int(len(x_train)/BATCH_SIZE)

    while True:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        np.random.shuffle(x_train)

        # Completion
        if sess.run(epoch) <= PRETRAIN_EPOCH:
            g_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                points_batch, mask_batch = get_points()

                _, g_loss = sess.run([g_train_op, model.g_loss],
                                     feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

            print('Completion loss: {}'.format(g_loss_value))

            np.random.shuffle(x_test)
            x_batch = x_test[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)
            if sess.run(epoch) == PRETRAIN_EPOCH:
                saver.save(sess, './backup/pretrained', write_meta_graph=False)


        # Discrimitation
        else:
            g_loss_value = 0
            d_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                points_batch, mask_batch = get_points()

                _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion],
                                                 feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

                local_x_batch = []
                local_completion_batch = []
                for i in range(BATCH_SIZE):
                    x1, y1, x2, y2 = points_batch[i]
                    local_x_batch.append(x_batch[i][y1:y2, x1:x2, :])
                    local_completion_batch.append(completion[i][y1:y2, x1:x2, :])
                local_x_batch = np.array(local_x_batch)
                local_completion_batch = np.array(local_completion_batch)

                _, d_loss = sess.run(
                    [d_train_op, model.d_loss],
                    feed_dict={x: x_batch, mask: mask_batch, local_x: local_x_batch, global_completion: completion,
                               local_completion: local_completion_batch, is_training: True})
                d_loss_value += d_loss

            print('Completion loss: {}'.format(g_loss_value))
            print('Discriminator loss: {}'.format(d_loss_value))

            np.random.shuffle(x_test)
            x_batch = x_test[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)


#定位局部区域
def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h

        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)



if __name__=='__main__':
    train()


