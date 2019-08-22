import os, pprint, time, h5py
import numpy as np
import tensorflow as tf
import tensorlayer as tl
pp = pprint.PrettyPrinter()
from utils_realAugm import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from nets2 import vgg2
import tensorflow.contrib.slim as slim
## Import for demo video
import imutils
from imutils.video import VideoStream
import cv2
import time

flags = tf.app.flags
flags.DEFINE_integer("id_classes", 15, "The number of subject categories")
flags.DEFINE_integer("anti_classes", 2, "The number of spoofing categories")
flags.DEFINE_float("lam", 0.3, "weights of the pcLoss")
flags.DEFINE_integer("output_size", 224, "the height of images")
flags.DEFINE_integer("batch_size", 32, "The number of batch images [64]")
#flags.DEFINE_string("antiType", "crop_backgrd_mfsdPhoto", "the types of for antispoofing- raw/rpyPhotoAdv/rpyPhotoCon ...")
flags.DEFINE_string("tstPath", "../FDA_codes/Replay-Attack/rpy_2_mfsdStyle/rpy_2_mfsdStyle_test.txt", "testing data path")
flags.DEFINE_string("lblIndx", "./id_labels/mfsd_ids.txt", "lables index path")
flags.DEFINE_string("pre_model", "../models/models/mfsd2rpy/tpc_mfsdPhoto_realAugm3-3/3_model.ckpt", "pretrained weights") # The model is trained on dataset MFSD
flags.DEFINE_boolean("is_resize", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

# fbl = open(FLAGS.lblIndx)
# files = fbl.readlines()
# lblDic = {}
# for xx in files:
#     if(xx.split(' ')[2] == 'train'):
#         ldc = {xx.split(' ')[0]:xx.split(' ')[3]}
#         lblDic.update(ldc)
 
def obtain_testAccuracy(sess,acc_id,acc_anti,probs_anti, preLbl_anti, input_x_id, input_y_id, input_x_anti, input_y_anti, tstPath):

    f = open(tstPath)
    data_files = [ os.path.join('..', os.curdir, line[35:]) for line in f.readlines()]
    f.close()
    data_files = data_files[0::1]
    shuffle(data_files)
    batch_idxs = len(data_files) // FLAGS.batch_size

    test_accID = 0
    test_accAnti = 0
    test_cost = 0
    grdLbls = []
    preLbls = []
    probas = []

    for idx in range(0, batch_idxs):
        batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
        # batch_files_anti = [v.replace('crop_backgrd', FLAGS.antiType) for v in batch_files]

            
        batch_id = [get_tst_image(batch_file, is_resize=FLAGS.is_resize, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
        batch_labels_id = [get_idxxx(batch_file, FLAGS.id_classes) for batch_file in batch_files]
        batch_images_id = np.array(batch_id).astype(np.float32)
            
        batch_anti = [get_tst_image(batch_file, is_resize=FLAGS.is_resize, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
        batch_labels_anti = [get_tst_antiLabel(batch_file, FLAGS.anti_classes) for batch_file in batch_files]
        batch_images_anti = np.array(batch_anti).astype(np.float32)
        
        tAccID, tAccAnti, pb, pl, gl = sess.run([acc_id, acc_anti, probs_anti, preLbl_anti,input_y_anti],feed_dict={input_x_id:batch_images_id, 
                                                   input_y_id:batch_labels_id, input_x_anti:batch_images_anti, input_y_anti:batch_labels_anti})
        pl = np.int64(pl)
        test_accID = test_accID + tAccID
        test_accID_avg = test_accID/(idx+1)
        test_accAnti = test_accAnti + tAccAnti
        test_accAnti_avg = test_accAnti/(idx+1)
        preLbls = np.append(preLbls, pl)
        grdLbls = np.append(grdLbls, gl)
        probas = np.append(probas, pb)

        if idx % 1 == 0:
            fp = open('../res/mfsd_2_rpy/print.txt', 'r+')
            print('**Step %d, tAccID = %.4f, tAccAnti = %.4f%% **'%(idx,test_accID_avg,test_accAnti_avg))
            print(gl)
            print(pl)
            print('**Step %d, tAccID = %.4f, tAccAnti = %.4f%% **'%(idx,test_accID_avg,test_accAnti_avg),file=(fp))
            print(gl, file=fp)
            print(pl, file=fp)
            fp.close()
    return test_accAnti_avg, grdLbls, preLbls, probas

def anti_demo(image, sess, acc_anti,probs_anti, preLbl_anti, input_x_id, input_y_id, input_x_anti, input_y_anti):

    probas = 0
    pl = None

    batch_id = [get_tst_image_for_demo(image, is_resize=FLAGS.is_resize, resize_w=FLAGS.output_size, is_grayscale = 0)]
    batch_labels_id = ['0']
    batch_images_id = np.array(batch_id).astype(np.float32)
        
    batch_anti = [get_tst_image_for_demo(image, is_resize=FLAGS.is_resize, resize_w=FLAGS.output_size, is_grayscale = 0)]
    batch_labels_anti = ['0']
    batch_images_anti = np.array(batch_anti).astype(np.float32)
    
    tAccAnti, pb, pl, gl = sess.run([acc_anti, probs_anti, preLbl_anti,input_y_anti],feed_dict={input_x_id:batch_images_id, 
                                                input_y_id:batch_labels_id, input_x_anti:batch_images_anti, input_y_anti:batch_labels_anti})
    probas = pb
    
    return probas, pl

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    ##========================= DEFINE MODEL ===========================##
    input_x_id = tf.placeholder(tf.float32, [None, FLAGS.output_size, FLAGS.output_size, 3])
    input_x_anti = tf.placeholder(tf.float32, [None, FLAGS.output_size, FLAGS.output_size, 3])
    input_y_id = tf.placeholder(tf.int64, shape=[None, ], name='y_id_grdth')
    input_y_anti = tf.placeholder(tf.int64, shape=[None, ], name='y_anti_grdth')
    
    # with slim.arg_scope(vgg2.vgg_arg_scope()):
        # net_simaese_id, end_points1_id = vgg2.vgg_siamese(input_x_id) 
    with slim.arg_scope(vgg2.vgg_arg_scope()):
        net_simaese_anti, end_points1_anti = vgg2.vgg_siamese(input_x_anti) 
    # net_id, end_points_id = vgg2.vgg_id(net_simaese_id, num_classes=FLAGS.id_classes, is_training=False) 
    net_anti, end_points_anti = vgg2.vgg_anti(net_simaese_anti, num_classes=FLAGS.anti_classes, is_training=False) 
    
    # y_id = tf.reshape(net_id, [-1, FLAGS.id_classes])
    y_anti = tf.reshape(net_anti, [-1, FLAGS.anti_classes])
    probs_anti = tf.nn.softmax(y_anti) 

    # correct_prediction_id = tf.equal(tf.cast(tf.argmax(y_id, 1), tf.float32), tf.cast(input_y_id, tf.float32))
    # acc_id = tf.reduce_mean(tf.cast(correct_prediction_id, tf.float32))
    # preLbl_id = tf.cast(tf.argmax(y_id, 1), tf.float32)
    
    correct_prediction_anti = tf.equal(tf.cast(tf.argmax(y_anti, 1), tf.float32), tf.cast(input_y_anti, tf.float32))
    acc_anti = tf.reduce_mean(tf.cast(correct_prediction_anti, tf.float32))
    preLbl_anti = tf.cast(tf.argmax(y_anti, 1), tf.float32)

    variables = slim.get_model_variables()

    ##========================= Test MODELS ================================##
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver_restore = tf.train.Saver(variables)
    saver_restore.restore(sess, FLAGS.pre_model)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
                
#    [test_accAnti, grdLbls, preLbls, probas] = obtain_testAccuracy(sess,acc_id,acc_anti,probs_anti, preLbl_anti,input_x_id, 
#                                                                input_y_id, input_x_anti, input_y_anti, FLAGS.tstPath)
    # [test_accAnti, grdLbls, preLbls, probas] = obtain_testAccuracy(sess,acc_id,acc_anti, probs_anti, preLbl_anti, input_x_id, 
    #                                                             input_y_id, input_x_anti, input_y_anti, FLAGS.tstPath)

    deploy_video_demo(sess, acc_anti, probs_anti, preLbl_anti, input_x_id, input_y_id, input_x_anti, input_y_anti, sconfidence=0.5)

def deploy_video_demo(sess, acc_anti, probs_anti, preLbl_anti, input_x_id, 
                                                                input_y_id, input_x_anti, input_y_anti, face_detector='../face_detector', fconfidence=0.5, sconfidence=0.5):
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([face_detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([face_detector, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting reading video ...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    skip = False

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels
        # grab the frame from the file
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        if not skip:
            
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > fconfidence:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX , startY, endX, endY) = box.astype("int")

                    # ensure the detected bounding box does fall outside the
                    # dimensions of the frame
                    startX = max(0, startX - 10)
                    startY = max(0, startY - 10)
                    endX = min(w, endX + 10)
                    endY = min(h, endY + 10)

                    # extract the face ROI and then preproces it in the exact
                    # same manner as our training data
                    face = frame[startY:endY, startX:endX]

                    # pass the face ROI through the trained liveness detector
                    # model to determine if the face is "real" or "fake"

                    probas, pl = anti_demo(face, sess, acc_anti, probs_anti, preLbl_anti, input_x_id, 
                                                                    input_y_id, input_x_anti, input_y_anti)
                    # probas_ = np.argmax(probas, 1)

                    # draw the label and bounding box on the frame)
                    label = 'REAL' if probas[0][0] > sconfidence else 'FAKE'
                    print(pl, probas, probas[0][0], label)
                    label = "{}:{:.4f}".format(label, probas[0][0 if label == 'REAL' else 1])
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        else:
            skip = not skip

        # show the output frame and wait for a key press
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    tf.app.run()

