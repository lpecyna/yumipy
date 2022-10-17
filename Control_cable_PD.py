from yumipy import YuMiRobot
import pyrealsense2 as rs
import numpy as np
import cv2
from time import sleep
from time import time
from multiprocessing import Process, Array, Value
from yumipy import YuMiState
import copy
import pickle
from autolab_core.rigid_transformations import RigidTransform


def pos_to_pixel(pose, a=[-361.38, 382.94], b=[1377.1, 1388.34]):
    if a is None or b is None:
        a = [-361.38, 382.94]
        b = [1377.1, 1388.34]
    p = pose.translation[0:2]
    x = round(a[0] + b[0] * p[0])
    y = round(a[1] + b[1] * p[1])
    return np.array([x, y])


def pixel_to_pos(pix, a=[-361.38, 382.94], b=[1377.1, 1388.34]):
    if a == None or b == None:
        a = [-361.38, 382.94]
        b = [1377.1, 1388.34]
    x = (pix[0] - a[0]) / b[0]
    y = (pix[1] - a[1]) / b[1]
    return np.array([x, y])


def data_from_cam(arr, pix_pos, num, dT):
    smoothing = 3
    VisibleV = np.zeros(smoothing)
    ConfidV = np.zeros(smoothing)
    AngV = np.zeros(smoothing)
    PosV = np.zeros(smoothing)

    # Size of rectangle where looking for rope
    hx = 30
    hy = 60
    # SET UP CAMERA:
    record = True
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
        width = 640 * 2 + hx * 2
        height = 480
        out = cv2.VideoWriter('video_PD_cable_following4_no_ang_cor.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    err_values = np.zeros(smoothing)
    j = 0
    i = 0
    # Start streaming
    pipeline.start(config)

    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        detected = 255 * np.ones([480, hx * 2]).astype('uint8')
        detected = cv2.cvtColor(detected, cv2.COLOR_GRAY2RGB)
        images = np.hstack((color_image, color_image, detected))
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
            color = (255, 0, 0)

            # Show point where approximated position:
            pix = [pix_pos[0], pix_pos[1]]
            edges = cv2.rectangle(edges, (pix[1] - 1, pix[0] - 1), (pix[1] + 1, pix[0] + 1), (100, 50, 255), 2)

            # Look for line - edge vertical:
            # Looking for near region that has edges (white) in vertical stripe of 2 pixels
            stripe_lenght = 70
            px = 0
            for i in range(20):
                sum_pix = np.sum(edges_nom[pix[0] - int(stripe_lenght):pix[0], pix[1] - 1 - i:pix[1] + 1 - i]) / 255
                # print(np.shape(edges_nom[pix[0] - int(stripe_lenght):pix[0], pix[1] - 1 - i:pix[1] + 1 - i]))
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
                sum_pix = np.sum(edges_nom[pix[0] - 1 - i:pix[0] + 1 - i, pix[1] - int(stripe_lenght):pix[1]]) / 255
                if sum_pix > 40:
                    py = -i
                    break
                sum_pix = np.sum(edges_nom[pix[0] - 1 + i:pix[0] + 1 + i, pix[1] - int(stripe_lenght):pix[1]]) / 255
                if sum_pix > 40:
                    py = i
                    break
            edges = cv2.rectangle(edges, (pix[1] - int(stripe_lenght), pix[0] - 2 + py), (pix[1], pix[0] + 1 + py),
                                  (0, 255, 255), 1)
            # frame - where look for points of rope:
            shift_right = 3
            extend = 20
            edges = cv2.rectangle(edges, (pix[1] - 1 + px + shift_right, pix[0] - 1 + py - hy),
                                  (pix[1] + px + hx + shift_right, pix[0] + py + extend), (255, 0, 0), 1)
            selected = gray[pix[0] + py - hy:pix[0] + py + extend,
                       pix[1] + px + shift_right:pix[1] + px + hx + shift_right]
            selected = np.where(selected < 70, 255, 0).astype('uint8')
            sum_rope = np.sum(selected, axis=0) / 255
            vis_length = 0
            # TODO This algorith for looking for the rope should be improoved, should look for continous smooth object
            # not just going right and look for amount of pixels
            for s in sum_rope:
                if s < 2:
                    break
                vis_length = vis_length + 1
            # Asssuming that there is no rope visible if vis_length<4
            if vis_length >= 4:
                visible = 1
                xy_coords = np.flip(np.column_stack(np.where(selected[:, 0:vis_length] >= 150)), axis=1)
                coef = np.polyfit(xy_coords[:, 0], xy_coords[:, 1], 1)
                ang = -np.arctan(coef[0])
                pos = 23 - 38 * (coef[1] - 7.0) / 50.5 - 3 # to shift a bit '3'
                #pos = pos - 34/2*np.tan(ang)
                #confid = vis_length / hx
            else:
                visible = 0
                pos = 0
                #confid = 0
                ang = 0
                coef = [0, 0]
            # Update smoothing arrays:
            VisibleV[0:-1] = VisibleV[1:]
            VisibleV[-1] = visible
            PosV[0:-1] = PosV[1:]
            PosV[-1] = pos
            #ConfidV[0:-1] = ConfidV[1:]
            #ConfidV[-1] = confid
            AngV[0:-1] = AngV[1:]
            AngV[-1] = ang
            if j < smoothing - 1:
                v = [visible, 0, ang, pos]
            else:
                # derivative - backward approximation from 3 last points
                d = (PosV[-3] / 2 - 2 * PosV[-2] + 3 / 2 * PosV[-1]) / dT
                d = visible*d
                #v = [max(VisibleV), (np.mean(ConfidV) + confid) / 2, (np.mean(AngV) + ang) / 2,
                #     (np.mean(PosV) + pos) / 2]
                v = [max(VisibleV), d, (np.mean(AngV) + ang) / 2, (np.mean(PosV) + pos) / 2]
            arr[:] = v
            if v[0] == 0:
                txt = "No Rope"
            else:
                txt = "Ang: {:.1f}, Pos.: {:.1f}, Conf: {:.1f}".format(v[2] * 180 / np.pi, v[3], v[1])

            # Show approximated line:
            selected = cv2.cvtColor(selected, cv2.COLOR_GRAY2RGB)
            selected = cv2.line(selected, [0, int(coef[1])], [vis_length, int(vis_length * coef[0]) + int(coef[1])],
                                (255, 0, 0))
            selected = cv2.resize(selected, [hx * 2, hy * 2])
            detected[240 - hy:240 + hy, :] = selected
            # Add text:
            org = (220, 50)
            edges = cv2.putText(edges, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
            # Show image:
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
    dT = 0.1
    __spec__ = None
    # SET UP PROCESS FOR CAM:
    arr = Array('d', [0, 0, 0, 0])
    num = Value('i', 0)
    pix_pos = Array('i', [210, 138])  # initial values - assumed
    p = Process(target=data_from_cam, args=(arr, pix_pos, num, dT))
    p.start()
    a = None
    b = None
    with open('Calibration_Copy.pkl', 'rb') as g:
        _, _, a, b = pickle.load(g)

    # SET UP ROBOT:
    # starting the robot interface

    y = YuMiRobot()

    y.left.set_speed(y.get_v(10))
    y.right.set_speed(y.get_v(50))
    y.left.set_zone(y.get_z('z0'))
    y.right.set_zone(y.get_z('z0'))
    # state_left_hand = YuMiState(vals=[-129.2, -127.9, -5.09, 100.7, -63.94, 85.62, 89.36])
    state_left_hand = YuMiState(vals=[-130.0, -125.05, -6.77, 99.91, -67.87, 84.44, 90.71])
    state_right_hand = YuMiState(vals=[57.25, -124.35, 19.05, 47.89, 34.82, -12.98, -75.36])
    y.left.goto_state(state_left_hand)
    y.right.goto_state(state_right_hand)
    pose_previous = y.left.get_pose()
    while arr[0] == 0:
        sleep(0.3)
    #sleep(60)

    for i in range(360):
        print(i)
        pose = y.left.get_pose()
        pix_pos[0:2] = pos_to_pixel(pose, a, b)
        #y.left.goto_pose_delta((np.random.randn() * 0.0004, 0.005, 0), wait_for_res=True)
        err = -arr[3]
        derr = -arr[1]
        print("err: ", err)
        print("derr: ", derr)
        # Velocities:
        # 10 - 10 mm/s
        # 0.01 step is 1 cm - 1s
        Vx = 0.005  # 10/s = 0.01 m/s
        mult = 1.5
        Kp = mult*0.00053  # 0.8*0.022*Vx#0.02 * Vx
        Kd = mult*0.00008  # 0.1*0.022*Vx*1.2#0.1*0.06*Vx*1.2#0.01 * Vx
        Vy = Kp * err + Kd * derr
        print("Vy:", Vy)
        V = np.sqrt(Vx * Vx + Vy * Vy)
        xy_step = [Vy * dT, Vx * dT]
        #print("current: ", xy_step)
        pose_previous.translation = pose_previous.translation + [xy_step[0], xy_step[1], 0]
        #pose_previous.translation = pose_previous.translation + [0, 0, 0]
        #print(round(V * 1000))
        y.left.set_speed(y.get_v(round(V * 1000)))
        y.left.goto_pose(pose_previous, wait_for_res=True)

    y.left.set_speed(y.get_v(50))
    y.left.goto_state(state_left_hand)
    y.stop()

    # pix_pos[0] = 210
    # pix_pos[1] = 138
    num.value = 1
    p.join()