import socket

while True:
    command = input("請輸入座標 (輸入 EXIT 結束): ")
    
    if command.upper() == "EXIT":
        break

    coords_list = [float(x) for x in command.split()]

    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9000))

    sock.sendall(str(coords_list).encode())

    #-120 -88 534 180 0 -92.5
    #

    sock.close()