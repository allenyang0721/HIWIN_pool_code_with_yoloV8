import numpy as np
import math

def map_value_x(x, in_min_x, in_max_x, out_max_x, out_min_x):
    return (x - in_min_x) * (out_max_x - out_min_x) / (in_max_x - in_min_x) + out_min_x

def map_value_y(y, in_min_y, in_max_y, out_min_y, out_max_y):
    return (y - in_min_y) * (out_max_y - out_min_y) / (in_max_y - in_min_y) + out_min_y

def angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    vector1_length = np.linalg.norm(vector1)
    vector2_length = np.linalg.norm(vector2)
    cos = dot_product / (vector1_length * vector2_length)
    arccos = np.arccos(cos)
    angle_degrees = np.degrees(arccos)
    return angle_degrees

def calculate_r_on_photo(cvx, cvy, center_x=640, center_y=320):
    return np.sqrt((center_x - cvx) ** 2 + (center_y - cvy) ** 2)

def first_calibration(x, y,angle_on_image):
    in_min_x = 128
    in_max_x = 1165
    out_min_x = -412
    out_max_x = 179

    in_min_y = 113
    in_max_y = 570
    out_min_y = -224
    out_max_y = 41

    a = map_value_x(x, in_min_x, in_max_x, out_min_x, out_max_x)
    b = map_value_y(y, in_min_y, in_max_y, out_min_y, out_max_y)
    #

    a_int = int(a)
    b_int = int(b)
    #
    c_int=angle_on_image
    c_ans=(88-c_int)-180 #計算出來的角度

    return a_int, b_int, 100, 180, 0, c_ans

# def second_calibration(old_x, old_y, cvx, cvy, c):
#     vec_1 = (cvx - 640, cvy - 320)
#     vec_2 = (640, 0)
#     k = angle(vec_1, vec_2)

#     l = calculate_r_on_photo(cvx, cvy)
#     k_rad = math.radians(k)

#     x_new = int(old_x - l * math.cos(k_rad) * 0.06040476197671861)
#     y_new = int(old_y - l * math.sin(k_rad) * 0.06040476197671861)
#     c_int = int(c)
#     c_ans = (88 - c_int) - 180

#     return x_new, y_new, 80, 180, 0, c_ans, k_rad

def second_calibration_new(a_int,b_int,sx,sy,angle_on_image):
    targetx,targety = 750,70#理想打擊點像素座標
    arm_movement = 12 #每移動一格機械手臂，會移動多少像素
    arm_x=(targetx-sx)
    arm_y=(targety-sy)
    theta=math.radians(angle_on_image)
    c_int=int(angle_on_image)
    c_ans=(88-c_int)-180 #計算出來的角度
    
    
    
    print(arm_x,arm_y)
    if arm_x > 0 and arm_y < 0:#1

        move_x=int(arm_x*math.cos(theta)-arm_y*math.sin(theta))
        move_y=int(-arm_y*math.cos(theta)-arm_x*math.sin(theta))
        print("case1")
    elif arm_x < 0 and arm_y <0:#2

        move_x=int(-arm_x*math.cos(theta)-arm_y*math.sin(theta))
        move_y=int(-arm_y*math.cos(theta)-arm_x*math.sin(theta))
        print("case2")
    elif arm_x < 0 and arm_y > 0:#3

        move_x=int(arm_x*math.cos(theta)+arm_y*math.sin(theta))
        move_y=int(arm_y*math.cos(theta)-arm_x*math.sin(theta))
        print("case3")
    elif arm_x > 0 and arm_y > 0:#4

        move_x=int(-arm_x*math.cos(theta)+arm_y*math.sin(theta))
        move_y=int(arm_y*math.cos(theta)+arm_x*math.sin(theta))
        print("case4")
    elif arm_x == 0 and arm_y < 0:#5

        move_x=int(-arm_y*math.sin(theta))
        move_y=int(-arm_y*math.cos(theta))
        print("case5")
    elif arm_x == 0 and arm_y > 0:#6

        move_x=int(arm_y*math.sin(theta))
        move_y=int(arm_y*math.cos(theta))
        print("case6")
    elif arm_x>0 and arm_y == 0:#7

        move_x=int(arm_x*math.cos(theta))
        move_y=int(-arm_x*math.sin(theta))
        print("case7")
    elif arm_x<0 and arm_y == 0:#8

        move_x=int(-arm_x*math.cos(theta))
        move_y=int(arm_x*math.sin(theta))
        print("case8")
    else:#9
        move_x=0
        move_y=0
        print("case9")
    print(f"move_x:{move_x},move_y:{move_y}")
    if move_x == 0 and move_y == 0:
        return a_int,b_int,100,180,0,c_ans
    else:
        a_move=a_int+(move_x/arm_movement)
        b_move=b_int+(move_y/arm_movement)
   

    return a_move,b_move,100,180,0,c_ans






def third_calibration(x, y, c_int, k_rad, push_distance_mm=50):
    push_distance_units = push_distance_mm * 1.0

    x_new = int(x - push_distance_units * math.cos(k_rad))
    y_new = int(y - push_distance_units * math.sin(k_rad))

    return x_new, y_new, 80, 180, 0, c_int

# === 主程式範例 ===
# 第一次校正
# x = 1101
# y = 292
# angle_on_image = -50

# a_int, b_int, _, _, _,_ = first_calibration(x, y)
# print("第一次校正座標:", a_int, b_int, 80, 180, 0, c_int)

# # 第二次校正
# cvx = 524
# cvy = 333

# a2_int, b2_int, _, _, _, c2_int, k_rad = second_calibration(a_int, b_int, cvx, cvy, c_int)
# print("第二次校正座標:", a2_int, b2_int, 80, 180, 0, c2_int)

# # 第三次校正
# a3_int, b3_int, _, _, _, c3_int = third_calibration(a2_int, b2_int, c2_int, k_rad)
# print("第三次校正座標:", a3_int, b3_int, 80, 180, 0, c3_int)

