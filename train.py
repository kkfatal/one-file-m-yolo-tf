# coding: utf-8

import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import PIL # pip install pillow
from PIL import Image, ImageDraw
#import cv2 # displaying

lr = 0.001 #0.0001 # learning rate
restore_at_begining = 0 # restore ANN model at begining of training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
clnum = 3*(5 + 2)
modelname = "checkpoint/model_bup.ckpt"
#pw = 7 # box max size
#ph = 7
dbg = 0 # display DEBUG info!
ktiram = 1 # keep training data in RAM 


if 0: # choose training data folder
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
    kernel = tf.Variable(tf.truncated_normal([4, 4, 150, 100],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[100], dtype=tf.float32), trainable=True, name='biases')
    conv5 = tf.nn.bias_add(conv, biases)
    conv5 = batch_norm(conv5, 100, tf.constant(True))
    conv5 = tf.nn.sigmoid(conv5, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)
  # conv6
  with tf.name_scope('conv6') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 100, 101],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[101], dtype=tf.float32), trainable=True, name='biases')
    conv6 = tf.nn.bias_add(conv, biases)
    conv6 = batch_norm(conv6, 101, tf.constant(True))
    conv6 = tf.nn.sigmoid(conv6, name=scope)
    parameters += [kernel, biases]
    print_activations(conv6)
  # conv7
  with tf.name_scope('conv7') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 101, 103],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[103], dtype=tf.float32), trainable=True, name='biases')
    conv7 = tf.nn.bias_add(conv, biases)
    conv7 = batch_norm(conv7, 103, tf.constant(True))
    conv7 = tf.nn.sigmoid(conv7, name=scope)
    parameters += [kernel, biases]
    print_activations(conv7)
  # conv8
  with tf.name_scope('conv8') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 103, 105],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[105], dtype=tf.float32), trainable=True, name='biases')
    conv8 = tf.nn.bias_add(conv, biases)
    conv8 = batch_norm(conv8, 105, tf.constant(True))
    conv8 = tf.nn.sigmoid(conv8, name=scope)
    parameters += [kernel, biases]
    print_activations(conv8)
  # conv9
  with tf.name_scope('conv9') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 105, clnum],dtype=tf.float32,stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[clnum], dtype=tf.float32), trainable=True, name='biases')
    conv9 = tf.nn.bias_add(conv, biases)
    conv9 = batch_norm(conv9, clnum, tf.constant(True))
    conv9 = tf.nn.sigmoid(conv9, name=scope)
    parameters += [kernel, biases]
    print_activations(conv9)

  return conv9, parameters

#num_class = 2
#sh = [1, 13, 13, 3*(5 + num_class) ]
ZEROSo = np.zeros([1, 29, 29, 21])
#yt = np.random.random_sample([1, 29, 29, 21])
print("yt shape : ", ZEROSo.shape)
ytrue = tf.placeholder(tf.float32, [1, 29, 29, 21])

mimages = tf.placeholder(tf.float32, [1,1000,1000,3])
imtmp = np.random.random_sample([1,1000,1000,3])

print("clnum : ", clnum)
pred, parameters = yolo_drx(mimages,clnum)

# Coding and decoding feature map!!!
#x = sigmoid(xo) + cx
#y = sigmoid(yo) + cy
#w = pw*exp(wo) 
#h = ph*exp(ho)
# Inverse
# xo= log(1/(x - cx)-1)  # sigmoid() [0,1] = 1.0 ./ ( 1.0 + exp(-z));[-inf; inf]
#y = sigmoid(yo) + cy
#w = pw*exp(wo) # 1 = log(exp(1))
#h = ph*exp(ho)
# feature map coding [obj1 = [ xo, yo, wo, ho, confid., class  ], obj2, obj3 ... ]
# feature_map shape (1, 29, 29, 21)

# GET DATA 
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
    if dbg:
        txt = fname+" "+root[1][0].text+" "+root[1][1].text+" "
        print("File name & size : ", txt)
    for size in root.findall("size"):
        #print ( "Decoding size ------->  ", float(object[1].text) )
        xsize = float(size[0].text)
        ysize = float(size[1].text)
        dx = xsize/29
        dy = ysize/29
    for obj in root.findall("object"): #"object"
        name = obj.find("name").text
        if name in code: 
            xmin = float(obj.find("bndbox").find("xmin").text)
            ymin = float(obj.find("bndbox").find("ymin").text)
            xmax = float(obj.find("bndbox").find("xmax").text)
            ymax = float(obj.find("bndbox").find("ymax").text)
            #print (" xmax  :  ", xmax, " object.tag : ", obj.tag)
            if dbg: 
                print( "xmin", xmin )
                print( "ymin", ymin )
                print( "xmax", xmax )
                print( "ymax", ymax )
            # Encode & decode (lots of versions)
            Nxn = int((xmax + xmin)/2/dx)
            if dbg: print( " Cell number x : ", Nxn)
            Cxn = ((xmax + xmin)/2 % dx )/dx
            #Cxn_i= -np.log(1/Cxn-1)
            Cxn_i = Cxn
            if dbg: print( " coeff. Cxn, Cxn_inv : ", Cxn, Cxn_i, 1/(1+np.exp(-Cxn_i))  )
            if dbg: print( " x center Cxn, centrs : ", int((xmax + xmin)/2), 1/(1 + np.exp(-Cxn_i))*dx + Nxn*dx )
            Nyn = int((ymax + ymin)/2/dy)
            if dbg: print( " Cell number y : ", Nyn)
            Cyn = ((ymax + ymin)/2 % dy )/dy
            #Cyn_i= -np.log(1/Cyn-1)
            Cyn_i = Cyn
            if dbg: print( " coeff. Cyn, Cyn_inv : ", Cyn, Cyn_i, 1/(1+np.exp(-Cyn_i)) )
            if dbg: print( " Y center Cyn, centrs : ", int((ymax + ymin)/2), 1/(1 + np.exp(-Cyn_i))*dy + Nyn*dy )
            #h = ph*exp(ho)
            #w = pw*exp(wo)
            W = xmax - xmin
            #w_i = np.log(W/pw) # w_i = W/pw
            # W = pw*wo # sigm [0,1]
            #w_i = np.log(W/dx)/pw
            #w_i = W/dx/pw
            #w_i = W/xsize
            w_i = (np.log(W*2/xsize)+3)/4
            H = ymax - ymin
            #h_i = np.log(H/ph) # h_i = H/ph 
            #h_i = np.log(H/dy)/ph
            #h_i = H/dy/ph
            #h_i = H/ysize
            h_i = (np.log(H*2/ysize)+3)/4
            if dbg: 
                print( " coeff. W, W_inv : ", W, w_i, pw*np.exp(w_i))
                print( " coeff. H, H_inv : ", H, h_i, ph*np.exp(h_i))
            # ZEROSo_c [1, 29, 29, 21]
            # [obj1 = [ xo, yo, wo, ho, confid., class  ], obj2, obj3 ... ]
            if ZEROSo_c[0,Nxn,Nyn, 4 ] < 0.01:
                ZEROSo_c[0,Nxn,Nyn, 0:5 ] = [Cxn_i, Cyn_i, w_i, h_i, 1.0]
                ZEROSo_c[0,Nxn,Nyn, 5:7 ] = code[name] #code[str(object[0].text)]
            elif ZEROSo_c[0,Nxn,Nyn, 4+7 ] < 0.01:
                ZEROSo_c[0,Nxn,Nyn, 7:12 ] = [Cxn_i, Cyn_i, w_i, h_i, 1.0]
                ZEROSo_c[0,Nxn,Nyn, 12:14 ] = code[name] #code[str(object[0].text)]
            else:
                ZEROSo_c[0,Nxn,Nyn, 14:19 ] = [Cxn_i, Cyn_i, w_i, h_i, 1.0]
                ZEROSo_c[0,Nxn,Nyn, 19:21 ] = code[name] #code[str(object[0].text)]
    return ZEROSo_c


# do tests on data gathering 
if 0:
    rn = int(np.random.random_sample(1)*len(annotations))
    ZEROSo_c = get_gt_feature_map(annotations[rn]) # Get groung true feature map
    im = Image.open(path +"/"+ annotations[rn][0:-4]+".jpg" ) # get associated image
    im = np.array( im.resize((1000, 1000), PIL.Image.ANTIALIAS) )
    im  = (im - im.min()) / (im.max() - im.min()) - 0.5
    #result = Image.fromarray((imtmp * 255).astype(numpy.uint8))
    #result.save('out2.jpg')
    #result.show()

def interpret_featuremap(fm):
    object_cnt = 0
    obj = []
    for n1 in range(29):
        for n2 in range(29):
            if fm[0,n1,n2,4]>0.1:
                cnfid = fm[0,n1,n2,4] # confidence
                #print("Object detected with confidence : ", cnfid )
                object_cnt = object_cnt + 1
                dx = 1000/29 
                dy = 1000/29 
                #x = 1/(1 + np.exp(-fm[0,n1,n2,0]))*dx + n1*dx   # xo ==> x = sigmoid(xo) + cx
                #y = 1/(1 + np.exp(-fm[0,n1,n2,1]))*dy + n2*dy   # yo ==> y = sigmoid(yo) + cy
                x = fm[0,n1,n2,0]*dx + n1*dx  
                y = fm[0,n1,n2,1]*dy + n2*dy
                #w = pw*np.exp(fm[0,n1,n2,2]) # #w = pw*exp(wo)
                #w = pw*fm[0,n1,n2,2]
                #w = np.exp(fm[0,n1,n2,2]*pw)*dx
                #w = fm[0,n1,n2,2]*pw*dx
                #w = fm[0,n1,n2,2]*1000
                w = np.exp(fm[0,n1,n2,2]*4-3)*1000/2 # inv. (np.log(W*2/1000)+3)/4
                #h = pw*np.exp(fm[0,n1,n2,3])
                #h = pw*fm[0,n1,n2,3]
                #h = np.exp(fm[0,n1,n2,3]*ph)*dy
                #h = fm[0,n1,n2,3]*ph*dy
                #h = fm[0,n1,n2,3]*1000
                h = np.exp(fm[0,n1,n2,3]*4-3)*1000/2
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
#loss_v2 = tf.reduce_sum((ytrue[:,:,:,0:2]-pred[:,:,:,0:2])*(ytrue[:,:,:,0:2]-pred[:,:,:,0:2])) + tf.reduce_sum((tf.sqrt(ytrue[:,:,:,2:4])-tf.sqrt(pred[:,:,:,2:4]))*(tf.sqrt(ytrue[:,:,:,2:4])-tf.sqrt(pred[:,:,:,2:4]))) + tf.reduce_sum((ytrue[:,:,:,4:]-pred[:,:,:,4:])*(ytrue[:,:,:,4:]-pred[:,:,:,4:]))

train = optimizer.minimize( loss )
#print ("Save/Restore the trained model !!!")
saver = tf.train.Saver() 


# training set for rapid trainig (keep all training data in RAM)
if ktiram:
    training_images = []
    feature_maps = []
    for rn in range(len(annotations)):
        im = Image.open(path +"/"+ annotations[rn][0:-4]+".jpg" ) # get associated image
        im = np.array( im.resize((1000, 1000), PIL.Image.ANTIALIAS) )
        im  = (im - im.min()) / (im.max() - im.min()) - 0.5
        training_images.append(im)
        ZEROSo_c = get_gt_feature_map(annotations[rn]) # Get groung true feature map
        feature_maps.append(ZEROSo_c)

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
            rn = int(np.random.uniform(0,1)*len(annotations))
            if ktiram: # 1 rapid training is on
                im = training_images[rn]
                ZEROSo_c = feature_maps[rn]
            else:
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
                if 0:
                    print ("Detected objects : ", obj)
                    print ("Ground truth objects : ", gt_obj)
                #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
                #cv2.resizeWindow("frame", 1000, 1000) 
                image = (im + 0.5)*255
                result = Image.fromarray((image).astype(np.uint8))
                for nkk in range(len(obj)): 
                    #if object_cnt > 0:
                    #    print("Detected label: ",code_i[np.argmax(obj[nkk][5])]," Ground truth label: ",code_i[np.argmax(gt_obj[nkk][5])])
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
                    #result = Image.fromarray((image).astype(np.uint8))
                    draw = ImageDraw.Draw(result)
                    #draw.rectangle(((0, 00), (100, 100)), fill="red")
                    draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),(xmin, ymin)], fill="blue", width=3)
                result.save('out.jpg')
                #cv2.imshow('frame',image)
                #cv2.waitKey()
                #print ("np.sum(featuremap) : ", np.sum(featuremap))
                #print (" row of confidence : ", featuremap[0,11,:,4] )
                #ZEROSo_c[0,Nxn,Nyn, 0:5 ]
                #[Cxn_i, Cyn_i, w_i, h_i, 1.0]
                #print (" Cxn_i : ", featuremap[0,:,:,0] )
                #print (" w_i : ", featuremap[0,:,:,2] )


            # save
            if step%5000 == 0:
                saver.save(sess, modelname )






print ("The End!")





