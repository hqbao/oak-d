#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")

# Linking
camLeft.out.link(xoutLeft.input)
camRight.out.link(xoutRight.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras:', device.getConnectedCameraFeatures())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    # Device name
    print('Device name:', device.getDeviceName(), ' Product name:', device.getProductName())

    # Output queue will be used to get the rgb frames from the output defined above
    qLeft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=2, blocking=False)

    while True:
        inLeft = qLeft.get()
        inRight = qRight.get()

        cv2.imshow("left", inLeft.getCvFrame())
        cv2.imshow("right", inRight.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
