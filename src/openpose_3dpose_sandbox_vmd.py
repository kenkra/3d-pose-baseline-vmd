#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# openpose_3dpose_sandbox_vmd.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import data_utils
import viz
import cameras
import os
from predict_3dpose import create_model
import cv2
import imageio
import logging
import datetime
import openpose_utils
import math
FLAGS = tf.app.flags.FLAGS

order = [15, 13, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

level = {0:logging.ERROR,
            1:logging.WARNING,
            2:logging.INFO,
            3:logging.DEBUG}

def show_anim_curves(anim_dict, _plt):
    val = np.array(list(anim_dict.values()))
    for o in range(0,36,2):
        x = val[:,o]
        y = val[:,o+1]
        logger.debug("x")
        logger.debug(x)
        logger.debug("y")
        logger.debug(y)
        _plt.plot(x, 'r--', linewidth=0.2)
        _plt.plot(y, 'g', linewidth=0.2)
    return _plt

def main(_):
    # 出力用日付
    now_str = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    logger.debug("FLAGS.person_idx={0}".format(FLAGS.person_idx))

    # 日付+indexディレクトリ作成
    subdir = '{0}/{1}_3d_{2}_idx{3:02d}'.format(os.path.dirname(openpose_output_dir), os.path.basename(openpose_output_dir), now_str, FLAGS.person_idx)
    os.makedirs(subdir)

    frame3d_dir = "{0}/frame3d".format(subdir)
    os.makedirs(frame3d_dir)

    #関節位置情報ファイル
    posf = open(subdir +'/pos.txt', 'w')

    #正規化済みOpenpose位置情報ファイル
    smoothedf = open(subdir +'/smoothed.txt', 'w')

    idx = FLAGS.person_idx - 1
    smoothed = openpose_utils.read_openpose_json(openpose_output_dir, idx, level[FLAGS.verbose] == 3)
    logger.info("reading and smoothing done. start feeding 3d-pose-baseline")
    logger.debug(smoothed)
    plt.figure(2)
    smooth_curves_plot = show_anim_curves(smoothed, plt)
    pngName = subdir + '/smooth_plot.png'
    smooth_curves_plot.savefig(pngName)
    
    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]

    actions = data_utils.define_actions(FLAGS.action)

    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    # before_pose = None
    device_count = {"GPU": 1}
    png_lib = []
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
        #plt.figure(3)
        batch_size = 128
        model = create_model(sess, actions, batch_size)

        # 以下の４つは仮の値で計算。多少違っていても、精度に影響はないと思う
        center_2d_x = 640 / 2 # x軸の中心（x解像度の半分）
        center_2d_y = 360 / 2 # y軸の中心（y解像度の半分）
        z_distance = 4000 # カメラから体までの距離(mm) 遠近の影響計算で使用
        camera_incline = 0 # カメラの水平方向に対する下への傾き（度）

        teacher_camera_incline = 13 # 教師データ(Human3.6M)のカメラの傾き（下向きに平均13度）
        # neckからHipまでが110ピクセル程度になるように入力を拡大する(教師データと同程度のスケール)
        input_scaling_factor = 110 / length_neck2hip_xy(smoothed)

        length_sum2d = 0
        length_sum3d = 0

        for n, (frame, xy) in enumerate(smoothed.items()):
            logger.info("calc idx {0}, frame {1}".format(idx, frame))

            # map list into np array  
            joints_array = np.zeros((1, 36))
            joints_array[0] = [0 for i in range(36)]
            for o in range(len(joints_array[0])):
                #feed array with xy array
                joints_array[0][o] = xy[o]
            _data = joints_array[0]

            smoothedf.write(' '.join(map(str, _data)))
            smoothedf.write("\n")

            # mapping all body parts or 3d-pose-baseline format
            for i in range(len(order)):
                for j in range(2):
                    # create encoder input
                    enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]
            for j in range(2):
                # Hip = (RHip + LHip) / 2
                enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                # Spine = (Thorax + Hip) / 2
                enc_in[0][12 * 2 + j] = (enc_in[0][0 * 2 + j] + enc_in[0][13 * 2 + j]) / 2

            # set spine
            # spine_x = enc_in[0][24]
            # spine_y = enc_in[0][25]

            # logger.debug("enc_in - 1")
            # logger.debug(enc_in)

            poses2d = enc_in
            # 入力データの拡大
            enc_in = enc_in * input_scaling_factor

            enc_in = enc_in[:, dim_to_use_2d]
            mu = data_mean_2d[dim_to_use_2d]
            stddev = data_std_2d[dim_to_use_2d]
            enc_in = np.divide((enc_in - mu), stddev)

            dp = 1.0
            dec_out = np.zeros((1, 48))
            dec_out[0] = [0 for i in range(48)]
            _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
            all_poses_3d = []
            enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
            poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
            plt.axis('off')
            all_poses_3d.append( poses3d )
            enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
            subplot_idx, exidx = 1, 1

            # 誤差を減らすため、OpenPose出力の(x, y)と3dPoseBaseline出力のzから、3次元の位置を計算する
            
            # OpenPose出力値とBaseline出力値のスケール比率を推定
            xy_scale, length_sum2d, length_sum3d = estimate_scale_xy(poses2d[0], poses3d[0], length_sum2d, length_sum3d)
            poses3d_op_xy = np.zeros((poses3d.shape[0], 96))
            for i in range(poses3d.shape[0]):
                for j in [0, 1, 2, 3, 6, 7, 8, 13, 15, 17, 18, 19, 25, 26, 27]:
                    # Hipとの差分
                    dy = poses3d[i][j * 3 + 1] - poses3d[i][0 * 3 + 1]
                    dz = poses3d[i][j * 3 + 2] - poses3d[i][0 * 3 + 2]
                    # 教師データのカメラ傾きを補正
                    dz = dz - dy * math.tan(math.radians(teacher_camera_incline - camera_incline))
                    # 遠近によるx,yの拡大率
                    z_ratio = (z_distance + dz) / z_distance
                    # x, yはOpenposeの値から計算
                    poses3d_op_xy[i][j * 3] = (poses2d[i][j * 2] - center_2d_x) * xy_scale * z_ratio
                    poses3d_op_xy[i][j * 3 + 1] = (poses2d[i][j * 2 + 1] - center_2d_y) * xy_scale * z_ratio
                    # zはBaselineの値から計算
                    poses3d_op_xy[i][j * 3 + 2] = dz

                # 12(Spine)はOpenPoseの出力にないため、代わりに13(Thorax)からの差分で計算する
                # 14(Neck/Nose)はOpenPoseの出力にないため、代わりに13(Thorax)からの差分で計算する
                # 15(Nose)はOpenPoseの出力にないため、代わりに13(Thorax)からの差分で計算する
                for j , k in [(12, 13),(14, 13),(15, 13)]:
                    # 差分
                    dx = poses3d[i][j * 3] - poses3d[i][k * 3]
                    dy = poses3d[i][j * 3 + 1] - poses3d[i][k * 3 + 1]
                    dz = poses3d[i][j * 3 + 2] - poses3d[i][k * 3 + 2]
                    # 教師データのカメラ傾きを補正
                    dz = dz - dy * math.tan(math.radians(teacher_camera_incline - camera_incline))
                    # x, y ,z
                    poses3d_op_xy[i][j * 3] = poses3d_op_xy[i][k * 3] + dx
                    poses3d_op_xy[i][j * 3 + 1] = poses3d_op_xy[i][k * 3 + 1] + dy
                    poses3d_op_xy[i][j * 3 + 2] = poses3d_op_xy[i][k * 3 + 2] + dz

            poses3d = poses3d_op_xy

            # 座標の変換
            poses3d = poses3d.reshape(-1, 3)
            R = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, -1, 0]
            ])
            poses3d = R.dot(poses3d.T)
            poses3d = poses3d.T
            poses3d = poses3d.reshape(-1, 96)

            # max = 0
            # min = 10000

            # logger.debug("enc_in - 2")
            # logger.debug(enc_in)

            # for i in range(poses3d.shape[0]):
            #     for j in range(32):
            #         tmp = poses3d[i][j * 3 + 2]
            #         poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
            #         poses3d[i][j * 3 + 1] = tmp
            #         if poses3d[i][j * 3 + 2] > max:
            #             max = poses3d[i][j * 3 + 2]
            #         if poses3d[i][j * 3 + 2] < min:
            #             min = poses3d[i][j * 3 + 2]

            # for i in range(poses3d.shape[0]):
            #     for j in range(32):
            #         poses3d[i][j * 3 + 2] = max - poses3d[i][j * 3 + 2] + min
            #         poses3d[i][j * 3] += (spine_x - 630)
            #         poses3d[i][j * 3 + 2] += (500 - spine_y)

            # Plot 3d predictions
            ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
            ax.view_init(18, 280)    
            logger.debug(np.min(poses3d))
            # if np.min(poses3d) < -1000 and before_pose is not None:
            #    poses3d = before_pose

            p3d = poses3d
            # logger.debug("poses3d")
            # logger.debug(poses3d)

            if level[FLAGS.verbose] == logging.INFO:
                viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71", add_labels=True, fix_range=True)

                # 各フレームの単一視点からのはINFO時のみ
                pngName = frame3d_dir + '/tmp_{0:012d}.png'.format(frame)
                plt.savefig(pngName)
                png_lib.append(imageio.imread(pngName))            
                # before_pose = poses3d

            # 各フレームの角度別出力はデバッグ時のみ
            if level[FLAGS.verbose] == logging.DEBUG:

                for azim in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
                    ax2 = plt.subplot(gs1[subplot_idx - 1], projection='3d')
                    ax2.view_init(18, azim)
                    viz.show3Dpose(p3d, ax2, lcolor="#FF0000", rcolor="#0000FF", add_labels=True, fix_range=True)

                    pngName2 = frame3d_dir + '/tmp_{0:012d}_{1:03d}.png'.format(frame, azim)
                    plt.savefig(pngName2)
            
            #関節位置情報の出力
            write_pos_data(poses3d, ax, posf)

        posf.close()

        # INFO時は、アニメーションGIF生成
        if level[FLAGS.verbose] == logging.INFO:
            logger.info("creating Gif {0}/movie_smoothing.gif, please Wait!".format(subdir))
            imageio.mimsave('{0}/movie_smoothing.gif'.format(subdir), png_lib, fps=FLAGS.gif_fps)

        logger.info("Done!".format(pngName))

# OpenPose出力値とBaseline出力値のスケール比率を推定する
# 2次元での骨格の長さ合計、3次元のxy平面での骨格の長さ合計の比率から推定
def estimate_scale_xy(channels2d, channels3d, length_sum2d, length_sum3d):
    assert channels3d.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels3d.size
    assert channels2d.size == len(data_utils.H36M_NAMES)*2, "channels should have 96 entries, it has %d instead" % channels2d.size
    vals3d = np.reshape( channels3d, (len(data_utils.H36M_NAMES), -1) )
    vals2d = np.reshape( channels2d, (len(data_utils.H36M_NAMES), -1) )

    I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
    J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points

    # xy平面上の骨格の長さの合計
    for i in np.arange( len(I) ): 
        length = np.sqrt((vals2d[I[i]][0] - vals2d[J[i]][0]) ** 2 + (vals2d[I[i]][1] - vals2d[J[i]][1]) ** 2)
        length_sum2d += length

        length = np.sqrt((vals3d[I[i]][0] - vals3d[J[i]][0]) ** 2 + (vals3d[I[i]][1] - vals3d[J[i]][1]) ** 2)
        length_sum3d += length
    
    if length_sum2d == 0:
        return 1, length_sum2d, length_sum3d

    return length_sum3d / length_sum2d, length_sum2d, length_sum3d

# 2次元でのNeckからHipまでの長さの平均を取得
def length_neck2hip_xy(smoothed):
    count = 0
    length = 0
    for (frame, xy) in smoothed.items():
        neck_x = xy[1 * 2]
        neck_y = xy[1 * 2 + 1]
        # Hip = RHip * 0.5 + LHip * 0.5
        hip_x = xy[8 * 2] * 0.5 + xy[11 * 2] * 0.5
        hip_y = xy[8 * 2 + 1] * 0.5 + xy[11 * 2 + 1] * 0.5
        length += ((neck_x - hip_x) ** 2 + (neck_y - hip_y) ** 2) ** 0.5 
        count += 1

    if count == 0 or length == 0:
        return 100
    
    return length / count

def write_pos_data(channels, ax, posf):

    assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

    I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
    J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
    # LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    #出力済みINDEX
    outputed = []

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        # for j in range(3):
        #     logger.debug("i={0}, j={1}, [vals[I[i], j]={2}, vals[J[i], j]]={3}".format(str(i), str(j), str(vals[I[i], j]), str(vals[J[i], j])))

        # 始点がまだ出力されていない場合、出力
        if I[i] not in outputed:
            # 0: index, 1: x軸, 2:Y軸, 3:Z軸
            logger.debug("I -> x={0}, y={1}, z={2}".format(x[0], y[0], z[0]))    
            posf.write(str(I[i]) + " "+ str(x[0]) +" "+ str(y[0]) +" "+ str(z[0]) + ", ")
            outputed.append(I[i])

        # 終点がまだ出力されていない場合、出力
        if J[i] not in outputed:
            logger.debug("J -> x={0}, y={1}, z={2}".format(x[1], y[1], z[1]))    
            posf.write(str(J[i]) + " "+ str(x[1]) +" "+ str(y[1]) +" "+ str(z[1]) + ", ")
            outputed.append(J[i])

        # xyz.append([x, y, z])

        # lines = ax.plot(x, y, z)
        # logger.debug("lines")
        # logger.debug(dir(lines))
        # for l in lines:
        #     logger.debug("l.get_data: ")
        #     logger.debug(l.get_data())
        #     logger.debug("l.get_data orig: ")
        #     logger.debug(l.get_data(True))
        #     logger.debug("l.get_path: ")
        #     logger.debug(l.get_path())

    #終わったら改行
    posf.write("\n")


if __name__ == "__main__":

    openpose_output_dir = FLAGS.openpose

    logger.setLevel(level[FLAGS.verbose])


    tf.app.run()