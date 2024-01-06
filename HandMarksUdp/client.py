import socket
import struct

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
client.bind(("", 37020))
while True:
    data, addr = client.recvfrom(1024)
    inum = struct.unpack('h', data[:2])
    findex = struct.unpack('I', data[2:6])
    landmarks = []
    for i in range(6, len(data), 9):
        data_byte = struct.unpack('B', data[i:i+1])
        x, y = struct.unpack('2f', data[i+1:i+9])
        landmarks.append((x, y, data_byte[0]))
    print(inum, findex, landmarks)