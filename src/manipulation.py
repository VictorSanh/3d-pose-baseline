from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import numpy.linalg as LA
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import xml.etree.ElementTree as ET

import procrustes
import viz
import cameras
import data_utils
import linear_model
import predict_3dpose

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--camera_frame", type=bool, default=False)
parser.add_argument("--max_norm", type=bool, default=False)
parser.add_argument("--batch_norm", type=bool, default=False)

parser.add_argument("--predict_14", type=bool, default=False)
parser.add_argument("--use_sh", type=bool, default=False)
parser.add_argument("--action", default="All")

parser.add_argument("--linear_size", type=int, default=1024)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--residual", type=bool, default=False)

parser.add_argument("--procrustes", type=bool, default=False)
parser.add_argument("--evaluateActionWise", type=bool, default=False)

parser.add_argument("--cameras_path", default="data/h36m/cameras.h5")
parser.add_argument("--data_dir", default="data/h36m/")
parser.add_argument("--train_dir", default="experiments")

parser.add_argument("--sample", type=bool, default=False)
parser.add_argument("--sample_specific", type=bool, default=False)
parser.add_argument("--use_cpu", type=bool, default=False)
parser.add_argument("--load", type=int, default=0)

parser.add_argument("--use_fp16", type=bool, default=False)

FLAGS, _ = parser.parse_known_args()

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

def getJ3dPosFromXML(XMLPath, nameDict=None):
    if nameDict is None:
        nameDict = {'R_Hip':0,
                    'R_Knee':1,
                    'R_Ankle':2,
                    'L_Hip':3,
                    'L_Knee':4,
                    'L_Ankle':5,
                    'L_Shoulder':6,
                    'L_Elbow':7,
                    'L_Wrist':8,
                    'R_Shoulder':9,
                    'R_Elbow':10,
                    'R_Wrist':11}
    annotation = ET.parse(XMLPath).getroot()
    keypoints = annotation.find('keypoints')
    GTPos = np.zeros((12,3))
    for keypoint in keypoints.findall('keypoint'):
        name = keypoint.get('name')
        x = float(keypoint.get('x'))
        y = float(keypoint.get('y'))
        # pay attention: convert to right hand coordinate frame by multiplying -1
        z = -1.*float(keypoint.get('z'))
        if name in nameDict.keys():
            GTPos[nameDict[name]] = np.array([x,y,z])
    return GTPos



def main(_):
    actions_all = data_utils.define_actions( "All" )

    # Load camera parameters
    SUBJECT_IDS = [1,5,6,7,8,9,11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

    # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions_all, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )
    print("Finished Read 3D Data")

    # _, _, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions_all, FLAGS.data_dir)
    _, _, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions_all, FLAGS.data_dir, rcams )
    print("Finished Read 2D Data")

    SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
    assert np.all( SH_TO_GT_PERM == np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]) )

    test_set = {}

    manipulation_dir = os.path.dirname(FLAGS.data_dir)
    manipulation_dir = os.path.dirname(manipulation_dir)
    manipulation_dir += '/manipulation_video/'
    manipulation_folders = glob.glob(manipulation_dir + '*')

    subj = 1
    action = 'manipulation-video'
    for folder in manipulation_folders:
        seqname = os.path.basename( folder )
        with h5py.File(folder + '/' + seqname +'.h5', 'r' ) as h5f:
            poses = h5f['poses'][:]

            # Permute the loaded data to make it compatible with H36M
            poses = poses[:,SH_TO_GT_PERM,:]

            # Reshape into n x (32*2) matrix
            poses = np.reshape(poses,[poses.shape[0], -1])
            poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])

            dim_to_use_x    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
            dim_to_use_y    = dim_to_use_x+1

            dim_to_use = np.zeros(len(SH_NAMES)*2,dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:,dim_to_use] = poses

            # If Uniframe Model Only keep the right 2d pose
            poses_final = poses_final[1::2,:]

            print(seqname, poses_final.shape)
            test_set[ (subj, action, seqname) ] = poses_final

    print("Finished Read Manipulations Videos")

    test_set_2d  = data_utils.normalize_data( test_set,  data_mean_2d, data_std_2d, dim_to_use_2d )
    dim_to_use_12_manipulation_joints = np.array([3,4,5, 6,7,8, 9,10,11, 18,19,20, 21,22,23, 24,25,26, 51,52,53, 54,55,56, 57,58,59, 75,76,77, 78,79,80, 81,82,83])


    print("Finished Normalize Manipualtion Videos")
    device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
    with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
        # === Create the model ===
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
        batch_size = FLAGS.batch_size #Intial code is 64*2
        model = predict_3dpose.create_model(sess, actions_all, batch_size)
        print("Model loaded")

        j = 0
        for key2d in test_set_2d.keys():

            (subj, b, fname) = key2d
            # if fname !=  specific_seqname + '.h5':
            #     continue
            print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )

            enc_in  = test_set_2d[ key2d ]
            n2d, _ = enc_in.shape

            # Split into about-same-size batches
            enc_in   = np.array_split( enc_in, n2d // 1 )
            all_poses_3d = []

            for bidx in range( len(enc_in) ):

                # Dropout probability 0 (keep probability 1) for sampling
                dp = 1.0
                anything = np.zeros((enc_in[bidx].shape[0], 48))
                _, _, poses3d = model.step(sess, enc_in[bidx], anything, dp, isTraining=False)

                # Denormalize
                enc_in[bidx]  = data_utils.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d )
                poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )
                all_poses_3d.append( poses3d )

            # Put all the poses together
            enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )


            enc_in, poses3d = map( np.vstack, [enc_in, poses3d] )

            poses3d_12_manipulation = poses3d[:, dim_to_use_12_manipulation_joints]

            annotated_images = glob.glob(manipulation_dir + fname + '/info/*.xml')
            annotated_images = sorted(annotated_images)



            # 1080p	= 1,920 x 1,080
            fig = plt.figure(j,  figsize=(10, 10))
            gs1 = gridspec.GridSpec(3, 3)
            gs1.update(wspace=-0, hspace=0.1) # set the spacing between axes.
            plt.axis('off')

            subplot_idx = 1
            nsamples = 3
            for i in np.arange( nsamples ):
                # Plot 2d Detection
                ax1 = plt.subplot(gs1[subplot_idx-1])
                img = mpimg.imread(manipulation_dir + fname + '/skeleton_cropped/' + os.path.basename(annotated_images[i]).split('_')[0] + '.jpg')
                ax1.imshow(img)

                # Plot 2d pose
                ax2 = plt.subplot(gs1[subplot_idx])
                p2d = enc_in[i,:]
                viz.show2Dpose( p2d, ax2 )
                ax2.invert_yaxis()

                # Plot 3d predictions
                # Compute first the procrustion and print error
                gt = getJ3dPosFromXML(annotated_images[i])
                A = poses3d_12_manipulation[i, :].reshape(gt.shape)
                _, Z, T, b, c = procrustes.compute_similarity_transform(gt, A, compute_optimal_scale=True)
                sqerr = np.sqrt(np.sum((gt - (b*A.dot(T)) - c)**2, axis = 1))
                print("{0} - {1} - Mean Error (mm) : {2}".format(fname, os.path.basename(annotated_images[i]), np.mean(sqerr)))

                ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
                temp = poses3d[i,:].reshape((32, 3))
                temp = c + temp.dot(T) #Do not scale
                p3d = temp.reshape((1, 96))
                viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )
                ax3.invert_zaxis()
                ax3.invert_yaxis()

                subplot_idx = subplot_idx + 3

            plt.show()
            j +=  1

if __name__ == "__main__":
    tf.app.run()
