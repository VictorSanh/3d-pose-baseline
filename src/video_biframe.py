"""Create a video of 3d predictions"""

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
import matplotlib.gridspec as gridspec
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import procrustes

import viz
import cameras
import data_utils
import linear_model_biframe
import predict_3dpose_biframe

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



def video():
  """Get samples from a model and visualize them"""

  actions_all = data_utils.define_actions( "All" )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions_all, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )
  train_set_3d = data_utils.remove_first_frame(train_set_3d)
  test_set_3d = data_utils.remove_first_frame(test_set_3d)
  train_root_positions = data_utils.remove_first_frame(train_root_positions)
  test_root_positions = data_utils.remove_first_frame(test_root_positions)
  print("Finished Read 3D Data")

  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions_all, FLAGS.data_dir)
  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.transform_to_2d_biframe_prediction(train_set_2d,
                                                                                                                                         test_set_2d,
                                                                                                                                         data_mean_2d,
                                                                                                                                         data_std_2d,
                                                                                                                                         dim_to_ignore_2d,
                                                                                                                                         dim_to_use_2d)
  print("Finished Read 2D Data")
  print(test_set_2d)

  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    batch_size = FLAGS.batch_size #Intial code is 64*2
    model = predict_3dpose_biframe.create_model(sess, actions_all, batch_size)
    print("Model loaded")

    for key2d in test_set_2d.keys():

      (subj, b, fname) = key2d
      # if subj != 11:
      #   continue
      # #if fname != 'Discussion 1.55011271.h5-sh':
      if (fname, subj)  not in [("Greeting 1.60457274.h5-sh", 9),
                                ("Photo.58860488.h5-sh", 9),
                                ("Directions 1.54138969.h5-sh", 9),
                                ("Purchases 1.55011271.h5-sh", 9),
                                ("Greeting.54138969.h5-sh", 11),
                                ("Discussion 1.55011271.h5-sh", 11),
                                ("Eating 1.55011271.h5-sh", 11),
                                ("Purchases 1.55011271.h5-sh", 11)]:
        continue
      print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )


      enc_in  = test_set_2d[ key2d ]
      n2d, _ = enc_in.shape
      print("Model Input has size : ", enc_in.shape)

      # Split into about-same-size batches
      enc_in   = np.array_split( enc_in, n2d // batch_size )
      all_poses_3d = []

      for bidx in range( len(enc_in) ):

        # Dropout probability 0 (keep probability 1) for sampling
        dp = 1.0
        anything = np.zeros((enc_in[bidx].shape[0], 48))
        _, _, poses3d = model.step(sess, enc_in[bidx], anything, dp, isTraining=False)

        # denormalize
        enc_in[bidx]  = data_utils.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d )
        poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )
        all_poses_3d.append( poses3d )

      # Put all the poses together
      enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )

      # Convert back to world coordinates
      if FLAGS.camera_frame:
        N_CAMERAS = 4
        N_JOINTS_H36M = 32


        cname = fname.split('.')[1]#camera_mapping[fname.split('.')[0][-1]] # <-- camera name "55011271"
        scams = {(subj,c+1): rcams[(subj,c+1)] for c in range(N_CAMERAS)} # cams of this subject
        scam_idx = [scams[(subj,c+1)][-1] for c in range(N_CAMERAS)].index( cname ) # index of camera used
        the_cam  = scams[(subj, scam_idx+1)] # <-- the camera used
        R, T, f, c, k, p, name = the_cam
        assert name == cname

        def cam2world_centered(data_3d_camframe):
          data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
          data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M*3))
          # subtract root translation
          return data_3d_worldframe - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS_H36M) )

        # Apply inverse rotation and translation
        poses3d = cam2world_centered(poses3d)

      # Grab a random batch to visualize
      enc_in, poses3d = map( np.vstack, [enc_in, poses3d] )


      #1080p	= 1,920 x 1,080
      fig = plt.figure( figsize=(7, 7) )
      gs1 = gridspec.GridSpec(1, 1)
      plt.axis('on')

      # dir_2d_poses = FLAGS.data_dir + 'S' + str(subj) + '/VideoBiframe/' + fname + '/2Destimate/'
      # if not os.path.isdir(dir_2d_poses):
      #   os.makedirs(dir_2d_poses)

      dir_3d_estimates = FLAGS.data_dir + 'S' + str(subj) + '/VideoBiframe/' + fname + '/3Destimate/'
      if not os.path.isdir(dir_3d_estimates):
        os.makedirs(dir_3d_estimates)


      # for i in np.arange( n2d ):
      #   # Plot 2d pose
      #   # ax1 = plt.subplot(gs1[0])
      #   # p2d = enc_in[i,:]
      #   # viz.show2Dpose( p2d, ax1 )
      #   # ax1.invert_yaxis()
      #   # fig.savefig(dir_2d_poses + str(i) + '.png')
      #
      #   # Plot 3d predictions
      #   ax3 = plt.subplot(gs1[0], projection='3d')
      #   p3d = poses3d[i,:]
      #   viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" , add_labels=True)
      #   fig.savefig(dir_3d_estimates + str(i+1) + '.png')
      #   if i==0:
      #     fig.savefig(dir_3d_estimates + str(i) + '.png')



def main(_):
  video()

if __name__ == "__main__":
  tf.app.run()
