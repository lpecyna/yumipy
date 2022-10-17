from yumipy import YuMiRobot
import pyrealsense2 as rs
import numpy as np
import cv2
from time import sleep
from time import time
from multiprocessing import Process, Array, Value
from yumipy import YuMiState
import copy
from autolab_core.rigid_transformations import RigidTransform
from msvcrt import getch
import pickle

COR_IMG = np.array([[185, 405], [95, 505]])
def calibration(cor_r):
    a = np.zeros([2])
    b = np.zeros([2])
    b[0] = (COR_IMG[0, 1] - COR_IMG[0, 0])/(cor_r[0, 1] - cor_r[0, 0])
    a[0] = COR_IMG[0, 0] - b[0]*cor_r[0, 0]
    b[1] = (COR_IMG[1, 1] - COR_IMG[1, 0]) / (cor_r[1, 1] - cor_r[1, 0])
    a[1] = COR_IMG[1, 0] - b[1] * cor_r[1, 0]
    return a, b


def pos_to_pixel(pose, a=[-361.38, 382.94], b=[1377.1, 1388.34]):
    if a is None or b is None:
        a = [-361.38, 382.94]
        b = [1377.1, 1388.34]
    p = pose.translation[0:2]
    x = round(a[0]+b[0]*p[0])
    y = round(a[1]+b[1]*p[1])
    return np.array([x, y])


def pixel_to_pos(pix, a=[-361.38, 382.94], b=[1377.1, 1388.34]):
    if a == None or b== None:
        a = [-361.38, 382.94]
        b = [1377.1, 1388.34]
    x = (pix[0]-a[0])/b[0]
    y = (pix[1]-a[1])/b[1]
    return np.array([x, y])


def data_from_cam(arr, pix_pos, num):
    smoothing = 3
    VisibleV = np.zeros(smoothing)
    ConfidV = np.zeros(smoothing)
    AngV = np.zeros(smoothing)
    PosV = np.zeros(smoothing)
    color = (255, 120, 80)
    #img_corners = [[185, 95],[405,95],[405,505][185,505]]
    #Size of rectange where looking for rope
    hx = 30
    hy = 60
    # SET UP CAMERA:
    record = False
    pipeline = rs.pipeline()
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if record:
        cap = cv2.VideoCapture(0)
        width = 640 * 2 + hx*2
        height = 480
        out = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    err_values = np.zeros(smoothing)
    j = 0
    # Start streaming
    pipeline.start(config)

    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        detected = 255*np.ones([480, hx*2]).astype('uint8')
        detected = cv2.cvtColor(detected, cv2.COLOR_GRAY2RGB)
        images = np.hstack((color_image, color_image,detected))
        cv2.imshow('RealSense', images)
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            gray = color_image[:, :, 2]  # chossing channal with highest contrast
            edges = cv2.Canny(gray, threshold1=30, threshold2=100)
            edges_nom = copy.copy(edges)

            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            txt = ""
            txt2 = "Bottom and right edges corner - move"
            if num.value == -1:
                txt = "Go to indicated left top and press: e"
                edges = cv2.rectangle(edges, (COR_IMG[1][0]-5, COR_IMG[0][0]-5), (COR_IMG[1][0]+5, COR_IMG[0][0]+5), (100, 50, 255), 2)
            if num.value == -2:
                txt = "Ok, saved, press: e, again"
                edges = cv2.rectangle(edges, (COR_IMG[1][0]-5, COR_IMG[0][0]-5), (COR_IMG[1][0]+5, COR_IMG[0][0]+5), (100, 50, 255), 2)
            if num.value == -3:
                txt = "Go to indicated left bottom and press: e"
                edges = cv2.rectangle(edges, (COR_IMG[1][0]-5, COR_IMG[0][1]-5), (COR_IMG[1][0]+5, COR_IMG[0][1]+5), (100, 50, 255), 2)
            if num.value == -4:
                txt = "Ok, saved, press: e, again"
                edges = cv2.rectangle(edges, (COR_IMG[1][0]-5, COR_IMG[0][1]-5), (COR_IMG[1][0]+5, COR_IMG[0][1]+5), (100, 50, 255), 2)
            if num.value == -5:
                txt = "Go to indicated right bottom and press: e"
                edges = cv2.rectangle(edges, (COR_IMG[1][1]-5, COR_IMG[0][1]-5), (COR_IMG[1][1]+5, COR_IMG[0][1]+5), (100, 50, 255), 2)
            if num.value == -6:
                txt = "Ok, saved, press: e, again"
                edges = cv2.rectangle(edges, (COR_IMG[1][1]-5, COR_IMG[0][1]-5), (COR_IMG[1][1]+5, COR_IMG[0][1]+5), (100, 50, 255), 2)
            if num.value == -7:
                txt = "Go to indicated right top and press: e"
                edges = cv2.rectangle(edges, (COR_IMG[1][1]-5, COR_IMG[0][0]-5), (COR_IMG[1][1]+5, COR_IMG[0][0]+5), (100, 50, 255), 2)
            if num.value == -8:
                txt = "Ok, saved, press: e, again"
                edges = cv2.rectangle(edges, (COR_IMG[1][1]-5, COR_IMG[0][0]-5), (COR_IMG[1][1]+5, COR_IMG[0][0]+5), (100, 50, 255), 2)
            if num.value == 4:
                #Show point where approximated position:
                pix = [pix_pos[0], pix_pos[1]]
                edges = cv2.rectangle(edges, (pix[1]-1, pix[0]-1), (pix[1] + 1, pix[0]+1), (100, 50, 255), 2)
                txt2 = "Test if bottom right edges detected"

                #Look for line - edge vertical:
                #Looking for near region that has edges (white) in vertical stripe of 2 pixels
                stripe_lenght = 70
                px = 0
                for i in range(20):
                    sum_pix = np.sum(edges_nom[pix[0] - int(stripe_lenght):pix[0], pix[1] - 1 - i:pix[1] + 1 - i]) / 255
                    #print(np.shape(edges_nom[pix[0] - int(stripe_lenght):pix[0], pix[1] - 1 - i:pix[1] + 1 - i]))
                    if sum_pix > 60:
                        px = -i
                        break
                    sum_pix = np.sum(edges_nom[pix[0] - int(stripe_lenght):pix[0], pix[1] - 1 + i:pix[1] + 1 + i]) / 255
                    if sum_pix > 60:
                        px = i
                        break
                edges = cv2.rectangle(edges, (pix[1] - 2 + px, pix[0] - int(stripe_lenght)), (pix[1] + 1 + px, pix[0]),
                                      (255, 255, 0), 1)

                # Look for line - edge horizontal:
                stripe_lenght = 60
                py = 0
                for i in range(20):
                    sum_pix = np.sum(edges_nom[pix[0]-1-i:pix[0]+1-i, pix[1] - int(stripe_lenght):pix[1]]) / 255
                    if sum_pix > 40:
                        py = -i
                        break
                    sum_pix = np.sum(edges_nom[pix[0]-1+i:pix[0]+1+i, pix[1] - int(stripe_lenght):pix[1]]) / 255
                    if sum_pix > 40:
                        py = i
                        break
                edges = cv2.rectangle(edges, (pix[1] - int(stripe_lenght), pix[0] - 2 + py), (pix[1], pix[0] + 1 + py),
                                      (0, 255, 255), 1)
                # frame - where look for points of rope:
                shift_right = 3
                extend = 20
                edges = cv2.rectangle(edges, (pix[1] - 1 + px+shift_right, pix[0] - 1 + py-hy), (pix[1] + px + hx + shift_right, pix[0] + py + extend), (255, 0, 0), 1)
                selected = gray[pix[0] + py-hy:pix[0] + py+extend, pix[1] + px+shift_right:pix[1] + px + hx + shift_right]
                selected = np.where(selected < 70, 255, 0).astype('uint8')
                sum_rope = np.sum(selected, axis=0)/255
                vis_length = 0
                # TODO This algorith for looking for the rope should be improoved, should look for continous smooth object
                # not just going right and look for amount of pixels
                for s in sum_rope:
                    if s < 4:
                        break
                    vis_length = vis_length + 1
                # Asssuming that there is no rope visible if vis_length<4
                if vis_length >= 4:
                    visible = 1
                    xy_coords = np.flip(np.column_stack(np.where(selected[:, 0:vis_length] >= 150)), axis=1)
                    coef = np.polyfit(xy_coords[:, 0], xy_coords[:, 1], 1)
                    ang = -np.arctan(coef[0])
                    pos = 23 - 38 * (coef[1] - 7.0) / 50.5
                    confid = vis_length/hx
                else:
                    visible = 0
                    pos = 0
                    confid = 0
                    ang = 0
                    coef = [0, 0]
                #Update smoothing arrays:
                VisibleV[0:-1] = VisibleV[1:]
                VisibleV[-1] = visible
                PosV[0:-1] = PosV[1:]
                PosV[-1] = pos
                ConfidV[0:-1] = ConfidV[1:]
                ConfidV[-1] = confid
                AngV[0:-1] = AngV[1:]
                AngV[-1] = ang
                if j < smoothing-1:
                    v = [visible, confid, ang, pos]
                else:
                    v = [max(VisibleV), (np.mean(ConfidV)+confid)/2, (np.mean(AngV)+ang)/2, (np.mean(PosV)+pos)/2]
                arr[:] = v
                if v[0] == 0:
                    txt = "No Rope"
                else:
                    txt = "Ang: {:.1f}, Pos.: {:.1f}, Conf: {:.1f}".format(v[2] * 180 / np.pi, v[3], v[1])

                #Show approximated line:
                selected = cv2.cvtColor(selected, cv2.COLOR_GRAY2RGB)
                selected = cv2.line(selected, [0, int(coef[1])], [vis_length, int(vis_length * coef[0]) + int(coef[1])],
                                 (255, 0, 0))
                selected = cv2.resize(selected, [hx * 2, hy * 2])
                detected[240-hy:240+hy,:] = selected

            #Add text:
            org = (150, 50)
            edges = cv2.putText(edges, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
            org = (150, 80)
            edges = cv2.putText(edges, txt2, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
            #Show image:
            images = np.hstack((color_image, edges))
            images = np.hstack((images, detected))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            j = j + 1
            if record:
                out.write(images)

            if cv2.waitKey(1) == 27 or num.value == 1:
                do_loop = False
                break
    finally:
        # Stop streaming
        pipeline.stop()
        if record:
            cap.release()
            out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    __spec__ = None
    print("Make sure robot gripper is oriented correctly and on a proper height (it was around 20 cm)")
    print("in the stand left hand values")
    #print("Do not run the server use jogging to move the arm to proper locations")
    corners = list(range(4))

    #SET UP PROCESS FOR CAM:
    arr = Array('d', [0, 0, 0, 0])
    num = Value('i', -1)
    pix_pos = Array('i', [210, 138]) # initial values - assumed
    p = Process(target=data_from_cam, args=(arr, pix_pos, num))
    p.start()
    y = YuMiRobot()
    y.left.set_speed(y.get_v(15))
    y.right.set_speed(y.get_v(50))
    y.left.set_zone(y.get_z('z0'))
    y.right.set_zone(y.get_z('z0'))
    # state_left_hand = YuMiState(vals=[-129.2, -127.9, -5.09, 100.7, -63.94, 85.62, 89.36])
    state_left_hand = YuMiState(vals=[-130.0, -125.05, -6.77, 99.91, -67.87, 84.44, 90.71])
    y.left.goto_state(state_left_hand)
    step = 0.0025
    a = None
    b = None
    while True:
        pose = y.left.get_pose()
        pix_pos[0:2] = pos_to_pixel(pose, a, b)
        #pix_pos[1] = pos_to_pixel(pose)[1]
        k = getch()
        if k == b'q':
            break
        elif k == b'w':
            y.left.goto_pose_delta((-step, 0, 0), wait_for_res=True)
        elif k == b'd':
            y.left.goto_pose_delta((0, step, 0), wait_for_res=True)
        elif k == b'a':
            y.left.goto_pose_delta((0, -step, 0), wait_for_res=True)
        elif k == b's':
            y.left.goto_pose_delta((step, 0, 0), wait_for_res=True)
        elif k == b'e':
            if num.value == -1:
                pose = y.left.get_pose()
                corners[0]=pose.translation[0:2]
                y.left.goto_pose_delta((0.14, 0, 0), wait_for_res=True)
            elif num.value == -3:
                pose = y.left.get_pose()
                corners[1] = pose.translation[0:2]
                y.left.goto_pose_delta((0, 0.28, 0), wait_for_res=True)
            elif num.value == -5:
                pose = y.left.get_pose()
                corners[2] = pose.translation[0:2]
                y.left.goto_pose_delta((-0.14, 0, 0), wait_for_res=True)
            elif num.value == -7:
                pose = y.left.get_pose()
                corners[3] = pose.translation[0:2]
                y.left.goto_pose_delta((0.07, -0.14, 0), wait_for_res=True)
                cor = np.array([[np.mean([corners[0][0],corners[3][0]]),np.mean([corners[1][0],corners[2][0]])],[np.mean([corners[0][1],corners[1][1]]),np.mean([corners[2][1],corners[3][1]])]])
                a, b = calibration(cor)
                with open('Calibration.pkl', 'wb') as f:
                    pickle.dump([cor, COR_IMG, a, b], f)
                    print("Calibration saved to Calibration.pkl")
                    print(a)
                    print(b)
                num.value = 5
            num.value = num.value - 1
        elif k == b'r':
            with open('Calibration.pkl', 'rb') as g:
                _, _, a, b = pickle.load(g)
            print(a)
            print(b)
            num.value = 4
            print("Test if edges correctly detected, 'q' - exit")

    y.left.set_speed(y.get_v(50))
    y.left.goto_state(state_left_hand)
    y.stop()
    num.value = 1
    p.join()
