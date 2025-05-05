import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
print("偵測到裝置數量：", len(devices))
for dev in devices:
    print("裝置名稱：", dev.get_info(rs.camera_info.name))
