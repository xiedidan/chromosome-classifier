import argparse
import struct

def read_jpeg(buffer, start):
    '''
    states = [
        'idle',
        'ff-start',
        'start',
        'ff-stop',
        'stop'
    ]
    '''

    state = 'idle'
    last_byte = 0xFF
    jpeg_buffer = []

    for i, byte in enumerate(buffer):
        if i < start:
            pass
        else:
            if state == 'idle' and byte == 0xFF:
                state = 'ff-start'
                last_byte = byte
            elif state == 'ff-start':
                if byte == 0xD8:
                    state = 'start'
                    jpeg_buffer.append(last_byte)
                    jpeg_buffer.append(byte)
                else:
                    state = 'idle'
                    jpeg_buffer.clear()
            elif state == 'start':
                if byte == 0xFF:
                    state = 'ff-stop'
                    last_byte = byte
                else:
                    jpeg_buffer.append(byte)
            elif state == 'ff-stop':
                if byte == 0xD9:
                    state = 'stop'
                    jpeg_buffer.append(last_byte)
                    jpeg_buffer.append(byte)
                    tail = i + 1
                    break
                else:
                    state = 'start'
                    jpeg_buffer.append(last_byte)
                    jpeg_buffer.append(byte)
    
    return jpeg_buffer, tail

if __name__ == '__main__':
    # arg
    parser = argparse.ArgumentParser(description='Image Extracting Util')
    parser.add_argument('--input', default='./sample.mmi', help='MMI file path')
    parser.add_argument('--output_pattern', default='{}-{}.jpg', help='output file pattern')
    flags = parser.parse_args()

    states = [
        'idle',
        'ff-start',
        'start',
        'ff-stop',
        'stop'
    ]

    state = 'idle'

    with open(flags.input, 'rb') as file:
        buffer = file.read()

    jpeg_buffer, tail = read_jpeg(buffer, 0)

    with open(flags.output_pattern.format('output', 0), 'wb') as file:
        for data in jpeg_buffer:
            byte = struct.pack('B', data)
            file.write(byte)

    jpeg_buffer, tail = read_jpeg(buffer, tail)

    with open(flags.output_pattern.format('output', 1), 'wb') as file:
        for data in jpeg_buffer:
            byte = struct.pack('B', data)
            file.write(byte)