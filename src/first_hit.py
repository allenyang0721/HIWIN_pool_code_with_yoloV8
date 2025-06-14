import socket
import time
import serial

home=[-120, -88, 534, 180, 0, -92.5]
first_hit = [-11,-99,100,180,0,-48]
second_hit = [-1,-91,100,180,0,-46]
third_hit = [-11,-85,76,180,0,-48]

def move_arm(coords_list):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9000))
    sock.sendall(str(coords_list).encode())
    time.sleep(1)
    print(f"手臂移動原點，座標：{coords_list}")
    sock.close()


move_arm(home)
time.sleep(4)
move_arm(first_hit)
time.sleep(2)
move_arm(second_hit)

move_arm(third_hit)
time.sleep(0.1)
# 3. 擊球（Arduino 控制）
ser = serial.Serial("COM6", 9600, timeout=1)
time.sleep(2)
ser.write(b'60' )#調整電壓!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ser.close()
print("✅ 發送擊球指令至 Arduino")
time.sleep(3)
move_arm(home)