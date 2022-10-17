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


def pos_to_pixel(pose):
    p = pose.translation[0:2]
    x = round(-361.38+1377.1*p[0])
    y = round(382.94+1388.34*p[1])
    return np.array([x, y])


def pixel_to_pos(pix):
    x = (pix[0]+361.38)/1377.1
    y = (pix[1]-382.94)/1388.34
    return np.array([x, y])


def data_from_cam(arr, pix_pos, num, dT):
    smoothing = 3
    # SET UP CAMERA:
    # Configure depth and color streams
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
        width = 640 * 2
        height = 480
        out = cv2.VideoWriter('video_PD_example_slow.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    err_values = np.zeros(smoothing)
    j = 0
    # Start streaming
    pipeline.start(config)

    try:
        t_old = time()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        images = np.hstack((color_image, color_image))
        cv2.imshow('RealSense', images)
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # if not depth_frame or not color_frame:
            #    continue
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            gray = color_image[:, :, 2]
            edges = cv2.Canny(gray, threshold1=30, threshold2=100)
            edges_nom = copy.copy(edges)
            top = 150  # 200
            bottom = 20#100  # 100
            left = 90
            right = 90
            # edges2 = edges[top:-bottom, left:-right]
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            color = (255, 0, 0)

            # frame:
            edges = cv2.rectangle(edges, (left, top), (640 - right, 480 - bottom), (0, 255, 0), 2)
            # frame - where look for points:
            pix = [pix_pos[0], pix_pos[1]]
            #print(pix)
            m = 55 # 20
            edges = cv2.rectangle(edges, (pix[1] + 7, max(top, pix[0] - m)), (pix[1] + 20, 480 - bottom), (0, 0, 255),
                                  2)
            edges2 = edges_nom[max(top, pix[0] - m):-bottom, pix[1] + 7:pix[1] + 20]
            xy_coords = np.flip(np.column_stack(np.where(edges2 >= 150)), axis=1)

            if np.size(xy_coords) > 0:
                r_xy = np.round(np.mean(xy_coords, 0) + [pix[1] + 7, max(top, pix[0] - m)]).astype(int)
                # +[top+30,pix[1]+7]
                #print(r_xy)
                edges = cv2.line(edges, [r_xy[0], r_xy[1]], [r_xy[0] + 10, r_xy[1]], (0, 255, 255), 2)
                err = r_xy[1] - pix[0]
                arr[2] = 1

                # coef = np.polyfit(xy_coords[:,0],xy_coords[:,1],1)
                # ang = -np.arctan(coef[0])
                # txt = "Angle = {:.1f} deg".format(ang*180/np.pi)
                txt = "Err in pixels: {:.1f}".format(err)
            else:
                # coef=np.array([0, 0])
                txt = "No object"
                err = 0
                arr[2] = 0
            err_values[0:-1] = err_values[1:]
            err_values[-1] = err
            if j < smoothing-1:
                arr[0] = err
                arr[1] = 0
            else:
                arr[0] = (np.mean(err_values)+err)/2 # the last added element is considered more important
                #derivative - backward approximation from 3 last points
                d = (err_values[-3]/2 - 2*err_values[-2] + 3/2*err_values[-1])/dT
                arr[1] = d*arr[2] # if not object not detected set derivative to 0
                #d2 = (-err_values[-2]+err_values[-1])/dT
                #print("derivative: ", d, ", or: ", d2)
            #arr[0] = err
            # text:
            org = (400, 50)
            edges = cv2.putText(edges, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)

            # ang_fin = ang_fin+ang/no_frames

            # else:
            images = np.hstack((color_image, edges))

            # Show images

            #t_n = time()
            #print("Time: ", t_n-t_old)
            #t_old = t_n
            #print("j: ", j)
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
    #SET UP PROCESS FOR CAM:
    arr = Array('d', [0, 0, 0])
    num = Value('i', 0)
    pix_pos = Array('i', [-1, -1])
    p = Process(target=data_from_cam, args=(arr, pix_pos, num, dT))
    p.start()

    # SET UP ROBOT:
    # starting the robot interface
    y = YuMiRobot()
    y.left.set_speed(y.get_v(20))
    y.right.set_speed(y.get_v(50))
    y.left.set_zone(y.get_z('z0'))
    y.right.set_zone(y.get_z('z0'))
    # y.left.set_zone(y.get_z('fine'))
    # y.right.set_zone(y.get_z('fine'))
    # getting the current pose of the right end effector
    # pose = y.left.get_pose()
    state_left_hand = YuMiState(vals=[-129.2, -127.9, -5.09, 100.7, -63.94, 85.62, 89.36])

    # Move to the start position:
    y.left.goto_state(state_left_hand)
    pose_previous = y.left.get_pose()
    # y.left.goto_pose_delta((0, 0.02, 0), wait_for_res=True)
    #sleep(10)
    for i in range(60):
        print(i)
        pose = y.left.get_pose()
        pix_pos[0] = pos_to_pixel(pose)[0]
        pix_pos[1] = pos_to_pixel(pose)[1]
        print("!!!!!!!!!!!!!!!!!!!!")
        #Move arm
        #step = 0.01
        # get pose (only first step - pose0)
        # pose1 = pose0
        # add translation to pose1 (easy - modify translation)
        # pose1
        # go to pose1

        # y.left.goto_pose_delta((-step*np.sin(ang_fin), step*np.cos(ang_fin), 0), wait_for_res=True)
        err = arr[0]
        derr = arr[1]
        print(arr[:])
        #K = 0.02#0.05
        #xy_step_prev = [K * step * err, step]
        #print("old step:", xy_step_prev)
        #y.left.goto_pose_delta((xy_step[0], xy_step[1], 0), wait_for_res=True)


        # Velocities:
        # 10 - 10 mm/s
        # 0.01 step is 1 cm - 1s
        Vx = 0.03#15  # 10/s = 0.01 m/s
        Kp = 0.00053#0.8*0.022*Vx#0.02 * Vx
        Kd = 0.00008#0.1*0.022*Vx*1.2#0.1*0.06*Vx*1.2#0.01 * Vx
        Vy = Kp * err + Kd * derr
        V = np.sqrt(Vx*Vx + Vy*Vy)
        xy_step = [Vy*dT, Vx*dT]

        print("current: ", xy_step)

        pose_previous.translation = pose_previous.translation + [xy_step[0], xy_step[1], 0]
        #y.left.goto_pose(pose_previous, velocity = V, wait_for_res=True)

        print(round(V*1000))
        y.left.set_speed(y.get_v(round(V*1000)))
        y.left.goto_pose(pose_previous, wait_for_res=True)


    #print("{:.1f}, {:.2f}, {:.2f}".format(ang*180/np.pi,np.sin(ang_fin), np.cos(ang_fin)))

    y.left.set_speed(y.get_v(50))
    y.left.goto_state(state_left_hand)
    y.stop()

    num.value = 1
    p.join()