"""my_lidar_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Radar


def run_robot(robot):
    # get the time step of the current world.
    timestep = 32
    #max speed of e-puck in webots
    max_speed = 6.28 
    
    #motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    #radar
    radar = robot.getDevice("radar")
    radar.enable(timestep)
    
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        # Read the sensors:
        num_targets = radar.getNumberOfTargets()
        targets = radar.getTargets()
        print("targets: {}".format(num_targets))
        for i in targets:
            print(vars(i))
        
    
        # Process sensor data here.
    
        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        left_motor.setVelocity(max_speed * 0.25)
        right_motor.setVelocity(max_speed * 0.25)
    
    # Enter here exit cleanup code.

if __name__ == "__main__":
    # create the Robot instance.
    my_robot = Robot()
    run_robot(my_robot)
