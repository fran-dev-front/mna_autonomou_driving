from controller import Robot

TIME_STEP = 64
robot = Robot()

#sensors
ds = []
dsNames = ['ds_right', 'ds_left']
for i in range(2):
    ds.append(robot.getDevice(dsNames[i]))
    ds[i].enable(TIME_STEP)
    
gyro = robot.getDevice("gyro")
gyro.enable(TIME_STEP)

imu = robot.getDevice("imu")
imu.enable(TIME_STEP)


#motors
wheels = []
wheelsNames = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
for i in range(4):
    wheels.append(robot.getDevice(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)


avoidObstacleCounter = 0
while robot.step(TIME_STEP) != -1:
    leftSpeed = 1.0
    rightSpeed = 1.0
    #gyro
    #print("gyro: {}".format(gyro.getValues()))
    """
    gyro_values = gyro.getValues()
    msg = "gyro: "
    for each_val in gyro_values:
        msg += "{0:0.5f}".format(each_val)
    print(msg)
    """
    imu_values = imu.getRollPitchYaw()
    msg = "imu: "
    for each_val in imu_values:
        msg += "{0:0.5f}".format(each_val)
    print(msg)
    
    if avoidObstacleCounter > 0:
        avoidObstacleCounter -= 1
        leftSpeed = 1.0
        rightSpeed = -1.0
    else:  # read sensors
        for i in range(2):
            #print("ds {}: {}".format(i,ds[i].getValue()))
            if ds[i].getValue() < 950.0:
                avoidObstacleCounter = 100
    wheels[0].setVelocity(leftSpeed)
    wheels[1].setVelocity(rightSpeed)
    wheels[2].setVelocity(leftSpeed)
    wheels[3].setVelocity(rightSpeed)
