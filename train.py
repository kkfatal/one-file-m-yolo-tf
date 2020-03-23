# coding: utf-8

import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import PIL # pip install Pillow
from PIL import Image
#import cv2 # just for displaying


lr = 0.001 #0.0001 # learning rate
restore_at_begining = 1 # restore ANN model at begining of training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
clnum = 3*(5 + 2)
modelname = "checkpoint/model_bup.ckpt"
pw = 90 # box max size
ph = 90
dbg = 0 # display DEBUG info!

if 0:
    path  = "./sm_images" # small image dataset
    #path  = "./t_images"
    path_an="./sm_annotations"
    #path_an="./t_annotations"
    code = {"kulturaugs":[1., 0.], "nezale":[0., 1.] }
    code_i = ["kulturaugs", "nezale" ]
else:
    path  = "./voc_images"
    path_an="./voc_annot"
    code = {"car":[1., 0.], "chair":[0., 1.] }
    code_i = ["car", "chair" ]



def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def yolo_drx(images, clnum):
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='biases')
    conv1 = tf.nn.bias_add(conv, biases)
    conv1 = batch_norm(conv1, 64, tf.constant(True))
    conv1 = tf.nn.sigmoid(conv1, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]
  pool1 = tf.nn.max_pool(conv1,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool1')
  print_activations(pool1)
  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),trainable=True, name='biases')
    conv2 = tf.nn.bias_add(conv, biases)
    conv2 = batch_norm(conv2, 192, tf.constant(True))
    conv2 = tf.nn.sigmoid(conv2, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)
  # pool2
  pool2 = tf.nn.max_pool(conv2,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool2')
  print_activations(pool2)
  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 201],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[201], dtype=tf.float32),trainable=True, name='biases')
    conv3 = tf.nn.bias_add(conv, biases)
    conv3 = batch_norm(conv3, 201, tf.constant(True))
    conv3 = tf.nn.sigmoid(conv3, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)
  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 201, 150],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[150], dtype=tf.float32), trainable=True, name='biases')
    conv4 = tf.nn.bias_add(conv, biases)
    conv4 = batch_norm(conv4, 150, tf.constant(True))
    conv4 = tf.nn.sigmoid(conv4, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)
  # pool4
  pool4 = tf.nn.max_pool(conv4,ksize=[1, 5, 5, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool5')
  print_activations(pool4)
  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 150, 100],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[100], dtype=tf.float32), trainable=True, name='biases')
    conv5 = tf.nn.bias_add(conv, biases)
    conv5 = batch_norm(conv5, 100, tf.constant(True))
    conv5 = tf.nn.sigmoid(conv5, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)
  # conv6
  with tf.name_scope('conv6') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 100, clnum],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[clnum], dtype=tf.float32), trainable=True, name='biases')
    conv6 = tf.nn.bias_add(conv, biases)
    conv6 = batch_norm(conv6, clnum, tf.constant(True))
    conv6 = tf.nn.sigmoid(conv6, name=scope)
    parameters += [kernel, biases]
    print_activations(conv6)

  return conv6, parameters

#num_class = 2
#sh = [1, 13, 13, 3*(5 + num_class) ]
ZEROSo = np.zeros([1, 29, 29, 21])
yt = np.random.random_sample([1, 29, 29, 21])
print("yt shape : ", ZEROSo.shape)
ytrue = tf.placeholder(tf.float32, [1, 29, 29, 21])

mimages = tf.placeholder(tf.float32, [1,1000,1000,3])
imtmp = np.random.random_sample([1,1000,1000,3])

print("clnum : ", clnum)
pred, parameters = yolo_drx(mimages,clnum)

#x = sigmoid(xo) + cx
#y = sigmoid(yo) + cy
#w = pw*exp(wo) # pw, ph are anchors
#h = ph*exp(ho)
# Inverse
# xo= log(1/(x - cx)-1)  # sigmoid() [0,1] = 1.0 ./ ( 1.0 + exp(-z));[-inf; inf]
#y = sigmoid(yo) + cy
#w = pw*exp(wo) # 1 = log(exp(1))
#h = ph*exp(ho)
  
# GET DATA 
# [obj1 = [ xo, yo, wo, ho, confid., class  ], obj2, obj3 ... ]
# get list of images 
allfiles = [f for f in listdir(path) if isfile(join(path, f))]
for iname in allfiles: 
    print(" Load image :  " + path +"/"+ iname)

# get list of annotations
annotations = [f for f in listdir(path_an) if isfile(join(path_an, f))] # list with file names
print("Files count = ", len(annotations) )

def get_gt_feature_map(fname):
    ZEROSo_c = ZEROSo.copy()
    #print ("ZEROSo_c  :  ", ZEROSo_c.shape)
    tree=ET.parse( path_an+"/"+fname)
    root=tree.getroot()
    #xsize = float(root[1][0].text)
    #dx = xsize / 29
    #ysize = float(root[1][1].text)
    #dy = ysize / 29
    if dbg:
        txt = fname+" "+root[1][0].text+" "+root[1][1].text+" "
        print("File name & size : ", txt)
    for object in root.findall('size'):
        #print ( "Decoding size ------->  ", float(object[1].text) )
        xsize = float(object[0].text)
        ysize = float(object[1].text)
        dx = xsize / 29
        dy = ysize / 29
    for object in root.findall('object'):
        #name = object.find('name').text
        if object.tag == "bndbox":
            xmin = float(object[0].text)
            ymin = float(object[1].text)
            xmax = float(object[2].text)
            ymax = float(object[3].text)
            if dbg: # Encode & decode
                print( "xmin", object[0].tag, xmin )
                print( "ymin", object[1].tag, ymin )
                print( "xmax", object[2].tag, xmax )
                print( "ymax", object[3].tag, ymax )
            #x = sigmoid(xo) + cx
            #y = sigmoid(yo) + cy
            #w = pw*exp(wo) # pw, ph are anchors
            #h = ph*exp(ho)
            # Inverse
            # xo= -log(1/(x - cx)-1)  # sigmoid() [0,1] = 1.0 ./ ( 1.0 + exp(-z));[-inf; inf]
            #y = sigmoid(yo) + cy
            #w = pw*exp(wo) # 1 = log(exp(1))
            #h = ph*exp(ho)
            Nxn = int((xmax + xmin)/2/dx)
            if dbg: print( " Cell number x : ", Nxn)
            Cxn = ((xmax + xmin)/2 % dx )/dx
            Cxn_i= -np.log(1/Cxn-1)
            if dbg: print( " coeff. Cxn, Cxn_inv : ", Cxn, Cxn_i, 1/(1+np.exp(-Cxn_i))  )
            if dbg: print( " x center Cxn, centrs : ", int((xmax + xmin)/2), 1/(1 + np.exp(-Cxn_i))*dx + Nxn*dx )

            Nyn = int((ymax + ymin)/2/dy)
            if dbg: print( " Cell number y : ", Nyn)
            Cyn = ((ymax + ymin)/2 % dy )/dy
            Cyn_i= -np.log(1/Cyn-1)
            if dbg: print( " coeff. Cyn, Cyn_inv : ", Cyn, Cyn_i, 1/(1+np.exp(-Cyn_i)) )
            if dbg: print( " Y center Cyn, centrs : ", int((ymax + ymin)/2), 1/(1 + np.exp(-Cyn_i))*dy + Nyn*dy )

            #h = ph*exp(ho)
            #w = pw*exp(wo)
            W = xmax - xmin
            #w_i = np.log(W/pw)
            # new version: W = pw*wo # sigm [0,1]
            w_i = W/pw
            H = ymax - ymin
            #h_i = np.log(H/ph)
            # new version
            h_i = H/ph
            if dbg: 
                print( " coeff. W, W_inv : ", W, w_i, pw*np.exp(w_i))
                print( " coeff. H, H_inv : ", H, h_i, ph*np.exp(h_i))
            # ZEROSo_c [1, 29, 29, 21]
            # [obj1 = [ xo, yo, wo, ho, confid., class  ], obj2, obj3 ... ]
            if ZEROSo_c[0,Nxn,Nyn, 4 ] < 0.01:
                ZEROSo_c[0,Nxn,Nyn, 0:5 ] = [Cxn_i, Cyn_i, w_i, h_i, 1.0]
                ZEROSo_c[0,Nxn,Nyn, 5:7 ] = code[str(object[0].text)]
            elif ZEROSo_c[0,Nxn,Nyn, 4+7 ] < 0.01:
                ZEROSo_c[0,Nxn,Nyn, 7:12 ] = [Cxn_i, Cyn_i, w_i, h_i, 1.0]
                ZEROSo_c[0,Nxn,Nyn, 12:14 ] = code[str(object[0].text)]
            else:
                ZEROSo_c[0,Nxn,Nyn, 14:19 ] = [Cxn_i, Cyn_i, w_i, h_i, 1.0]
                ZEROSo_c[0,Nxn,Nyn, 19:21 ] = code[str(object[0].text)]
    return ZEROSo_c


# do tests on data gathering 
rn = int(np.random.random_sample(1)*len(annotations))
ZEROSo_c = get_gt_feature_map(annotations[rn]) # Get groung true feature map
im = Image.open(path +"/"+ annotations[rn][0:-4]+".jpg" ) # get associated image
im = np.array( im.resize((1000, 1000), PIL.Image.ANTIALIAS) )
im  = (im - im.min()) / (im.max() - im.min()) - 0.5

#result = Image.fromarray((imtmp * 255).astype(numpy.uint8))
#result.save('out2.jpg')
#result.show()

# [obj1 = [ xo, yo, wo, ho, confid., class  ], obj2, obj3 ... ]
# feature_map (1, 29, 29, 21)
#x = sigmoid(xo) + cx
#y = sigmoid(yo) + cy
#w = pw*exp(wo) # pw, ph are anchors
#h = ph*exp(ho)
def interpret_featuremap(fm):
    object_cnt = 0
    obj = []
    for n1 in range(29):
        for n2 in range(29):
            if fm[0,n1,n2,4]>0.1:
                cnfid = fm[0,n1,n2,4] # confidence
                #print("Object detected with confidence : ", cnfid )
                object_cnt = object_cnt + 1
                dx = 1000 / 29 #np.floor(1000 / 29)
                dy = 1000 / 29 #np.floor(1000 / 29)
                x = 1/(1 + np.exp(-fm[0,n1,n2,0]))*dx + n1*dx   # xo ==> x = sigmoid(xo) + cx
                y = 1/(1 + np.exp(-fm[0,n1,n2,1]))*dy + n2*dy   # yo ==> y = sigmoid(yo) + cy
                #w = pw*np.exp(fm[0,n1,n2,2]) # #w = pw*exp(wo)
                # new version
                w = pw*fm[0,n1,n2,2]
                #h = pw*np.exp(fm[0,n1,n2,3])
                # new version 
                h = pw*fm[0,n1,n2,3]
                cl = fm[0,n1,n2,5:7]
                obj.append([x, y, w, h, cnfid, cl])
            if fm[0,n1,n2,4+7]>0.1:
                #print("Object detected with confidence : ", fm[0,n1,n2,7] )
                object_cnt = object_cnt + 1
            if fm[0,n1,n2,4+14]>0.1:
                #print("Object detected with confidence : ", fm[0,n1,n2,18] )
                object_cnt = object_cnt + 1
    return object_cnt, obj

optimizer = tf.train.GradientDescentOptimizer( lr ) 
print_activations(ytrue)
print_activations(pred)
loss = tf.reduce_sum((ytrue-pred)*(ytrue-pred))
# [obj1 = [ xo, yo, wo, ho, confid., class  ], obj2, obj3 ... ]
# feature_map (1, 29, 29, 21)
#loss_v2 = tf.reduce_sum((ytrue[:,:,:,0:2]-pred[:,:,:,0:2])*(ytrue[:,:,:,0:2]-pred[:,:,:,0:2])) + tf.reduce_sum((tf.sqrt(ytrue[:,:,:,2:4])-tf.sqrt(pred[:,:,:,2:4]))*(tf.sqrt(ytrue[:,:,:,2:4])-tf.sqrt(pred[:,:,:,2:4]))) + tf.reduce_sum((ytrue[:,:,:,4:]-pred[:,:,:,4:])*(ytrue[:,:,:,4:]-pred[:,:,:,4:]))

train = optimizer.minimize( loss )
#print ("Save/Restore the trained model !!!")
saver = tf.train.Saver() 

# TRAIN!!!
if 1:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # initialize
        #sess.run(tf.initialize_all_variables())
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if restore_at_begining:
            saver.restore(sess, modelname)
        for step in range(1000000):
            # gather data
            rn = int(np.random.random_sample(1)*len(annotations))
            ZEROSo_c = get_gt_feature_map(annotations[rn]) # Get groung true feature map
            im = Image.open(path +"/"+ annotations[rn][0:-4]+".jpg" ) # get associated image
            im = np.array( im.resize((1000, 1000), PIL.Image.ANTIALIAS) )
            im  = (im - im.min()) / (im.max() - im.min()) - 0.5
            # train
            __train, __loss = sess.run([train, loss ], feed_dict={mimages: [im], ytrue: ZEROSo_c})
            # report 
            if step%100 == 0:
                print("Step : ", step, " loss : ", __loss)
                # calculate feature map
                featuremap = sess.run( pred, feed_dict={mimages: [im], ytrue: ZEROSo_c})
                #print ("feature_map", feature_map.shape) feature_map (1, 29, 29, 21)
                object_cnt, obj = interpret_featuremap(featuremap)
                gt_object_cnt, gt_obj = interpret_featuremap(ZEROSo_c)
                print ("Detected object_cnt : ", object_cnt, " Ground truth obj_cnt : ", gt_object_cnt)
                print ("Detected objects : ", obj)
                print ("Ground truth objects : ", gt_obj)
                #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
                #cv2.resizeWindow("frame", 1000, 1000) 
                image = (im + 0.5)*255
                result = Image.fromarray((image).astype(np.uint8))
                for nkk in range(len(obj)): 
                    if object_cnt > 0:
                        print("Detected label: ",code_i[np.argmax(obj[nkk][5])]," Ground truth label: ",code_i[np.argmax(gt_obj[nkk][5])])
                    #print ("arguments : ", obj[nkk]) #[obj[nkk,0]-obj[nkk,2]/2,obj[nkk,1]-obj[nkk,3]/2])
                    xmin = int(obj[nkk][0]-obj[nkk][2]/2)
                    #print ("xmin : ", xmin ) #obj[nkk,0]-obj[nkk,2]/2) 
                    ymin = int(obj[nkk][1]-obj[nkk][3]/2)
                    #print ("ymin : ", ymin)
                    xmax = int(obj[nkk][0]+obj[nkk][2]/2)
                    #print ("xmax : ", xmax)
                    ymax = int(obj[nkk][1]+obj[nkk][3]/2) 
                    #print ("ymax : ", ymax)
                    #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                    #cv2.imwrite("out.jpg", image)
                    from PIL import ImageDraw
                    #result = Image.fromarray((image).astype(np.uint8))
                    draw = ImageDraw.Draw(result)
                    #draw.rectangle(((0, 00), (100, 100)), fill="red")
                    draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),(xmin, ymin)], fill=None, width=3)
                result.save('out.jpg')
                #cv2.imshow('frame',image)
                #cv2.waitKey()
                print ("np.sum(featuremap) : ", np.sum(featuremap))
                #print (" row of confidence : ", featuremap[0,11,:,4] )
                #ZEROSo_c[0,Nxn,Nyn, 0:5 ]
                #[Cxn_i, Cyn_i, w_i, h_i, 1.0]
                #print (" Cxn_i : ", featuremap[0,:,:,0] )
                #print (" w_i : ", featuremap[0,:,:,2] )


            # save
            if step%5000 == 0:
                saver.save(sess, modelname )






print ("The End!")







