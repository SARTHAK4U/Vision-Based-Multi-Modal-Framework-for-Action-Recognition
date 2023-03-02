import glob
import cv2
from pathlib import Path
import os
import cv2
import numpy as np


def get_dynamic_image(frames, normalized=True):
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def _get_channel_frames(iter_frames, num_channels):
    frames = [[] for channel in range(num_channels)]

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return frames


def _compute_dynamic_image(frames):
    num_frames, h, w, depth = frames.shape
    print(frames.shape)
    coefficients = np.zeros(num_frames)
    print('coeff =',coefficients)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        print('cumulative_indice : ',cumulative_indices)
        coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)
        print('coefficients[n] : ',n,coefficients[n])

    x1 = np.expand_dims(frames, axis=0)
    print('x1 : ',x1.shape)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    print('x2 : ',x2.shape)
    result = x1 * x2
    print(result.shape)
    return np.sum(result[0], axis=0).squeeze()


def get_video_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frame_list = []
    try:
        while True:
            more_frames, frame = video.read()

            if not more_frames:
                break
            else:
                frame_list.append(frame)
    finally:
        video.release()
    return frame_list

def get_video_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frame_list = []
    try:
        while True:
            more_frames, frame = video.read()

            if not more_frames:
                break
            else:
                frame_list.append(frame)

    finally:
        video.release()

    return frame_list


def main():
    dataDir = "./n"
    for file in os.listdir(dataDir):
    #   print(file)
      path="./example_frames/"+file
      frames=get_video_frames(path)
      dyn_image = get_dynamic_image(frames, normalized=True)
      cv2.imshow('', dyn_image)
    #   file=file.replace(".avi","")
    #   cv2.imwrite(file+".jpg", dyn_image)
      cv2.waitKey()

if __name__ == '__main__':
    main()
