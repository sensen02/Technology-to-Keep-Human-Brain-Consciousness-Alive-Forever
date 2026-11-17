"""Bounded reacquisition of the fixed red landmark from the actual eye frames."""
from collections import deque

import numpy as np
import torch

from approach_vision import features, network
from far_optics import WEIGHTS

FOV_SCALE = np.tan(np.deg2rad(50.)) / 16.
PITCH = np.deg2rad(-15.)
EYE_YAWS = np.deg2rad([25.,-25.])
RED_MIN = 9.


def marker_bearing(image):
    pixels = image.astype(float)
    weight = np.maximum(pixels[...,0] - .5*(pixels[...,1]+pixels[...,2]) - RED_MIN, 0.)
    angles = []
    rows, cols = np.indices((32,32)) - 15.5
    for signal, yaw in zip(weight, EYE_YAWS):
        mass = signal.sum()
        if not mass:
            continue
        x = FOV_SCALE * float(np.sum(signal * cols) / mass)
        y = -FOV_SCALE * float(np.sum(signal * rows) / mass)
        forward = np.array([np.cos(PITCH)*np.cos(yaw), np.cos(PITCH)*np.sin(yaw), np.sin(PITCH)])
        up = np.array([-np.sin(PITCH)*np.cos(yaw),-np.sin(PITCH)*np.sin(yaw),np.cos(PITCH)])
        direction = forward + x * np.cross(forward,up) + y * up
        angles.append(np.rad2deg(np.arctan2(direction[1],direction[0])))
    return float(np.mean(angles)) if angles else None


class VisualRejoin:
    def __init__(self):
        self.model = network().eval()
        self.model.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True))
        self.phase = "align"
        self.scan = 0
        self.scan_sign = -1
        self.timer = 0
        self.near = False
        self.stop_confirm = 0
        self.estimate = deque(maxlen=5)

    @torch.inference_mode()
    def act(self, image):
        raw_angle = marker_bearing(image)
        if raw_angle is None:
            self.estimate.clear()
            self.stop_confirm = 0
            prediction = (0.,0.,-1.)
        else:
            angle, distance, confidence = self.model(features(torch.as_tensor(image[None])))[0].tolist()
            angle *= 40.
            if confidence < 0. or abs(angle-raw_angle) > 9.:
                angle = raw_angle
            self.estimate.append((angle,distance+2.5))
            angle,distance = np.median(self.estimate,axis=0)
            prediction = (float(angle),float(distance),float(confidence))
            if abs(angle)>3.:
                self.scan_sign = 1 if angle>0 else -1

        if self.phase == "done":
            return (0.,0), prediction
        if self.phase == "settle":
            self.timer -= 1
            if self.timer > 0:
                return (0.,0), prediction
            self.phase = "walk" if raw_angle is not None and abs(prediction[0])<=10. else "align"

        if self.phase == "align":
            if raw_angle is None:
                self.scan += 1
                if self.scan <= 60:
                    return (0.,self.scan_sign), prediction
                if self.scan <= 160:
                    return (0.,-self.scan_sign), prediction
                if self.scan <= 270:
                    return (0.,self.scan_sign), prediction
                return (0.,0), prediction
            self.scan = 0
            if abs(prediction[0])>6.:
                return (0.,self.scan_sign), prediction
            self.phase = "settle"
            self.timer = 40
            return (0.,0), prediction

        if raw_angle is None or abs(prediction[0])>16.:
            self.phase = "settle"
            self.timer = 40
            return (0.,0), prediction
        distance = prediction[1]
        self.stop_confirm = self.stop_confirm + 1 if prediction[2] >= 0. and distance < 1.46 else 0
        if self.stop_confirm >= 5:
            self.phase = "done"
            return (0.,0), prediction
        self.near = self.near or distance < 2.41
        return (.071713 if self.near else .08318,0), prediction
