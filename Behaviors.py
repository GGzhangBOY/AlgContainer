# THIS FILE IS PART OF UE4 LIDAR SIMULATE AND AUTO DRIVE SIMULATE PROJECT 
# THIS PROGRAM IS FREE SOFTWARE, IS LICENSED UNDER MIT
# Copyright(c) Bowei Zhang
# https://github.com/GGzhangBOY/AlgContainer
import DataReceiver

class SteeringWheel:
    def __init__(self, in_delay_time = 2):
        self.currentAngle = 0
        self.delay_time = in_delay_time
        self.saved_turing_offset = 0
        self.current_delay_time = 0

    def turnTo(self, angle):
        
        if(abs(angle) <= 3):
            return -self.currentAngle

        if(self.current_delay_time is 0):
            self.saved_turing_offset = angle/self.delay_time
            self.current_delay_time = self.delay_time
        self.currentAngle += self.saved_turing_offset
        self.current_delay_time -= 1
        return self.saved_turing_offset

class SteeringMechanism:

    def __init__(self, in_SW_delay_time = 2):
        self.steeringWheelComponent = SteeringWheel(in_SW_delay_time)

    def steeringWheelRequest(self, angle = 0):
        self.steeringWheelComponent.turnTo(angle)

    def returnSteeringResult(self):
        return self.steeringWheelComponent.saved_turing_offset


class Brake:
    def __init__(self):
        self.BreakPersent = 0
    
    def breakTo(self, BreakingRate = 0):
        self.BreakingRate = BreakingRate

class BrakingMechanism:
    def __init__(self):
        self.brakingComponent = Brake()
    
    def brakeRequest(self, in_BrakingRate):
        self.brakingComponent.BreakPersent = in_BrakingRate

    def returnBrakingResult(self):
        return self.brakingComponent.BreakPersent




class Throttle:
    def __init__(self):
        self.ThrottlePersent = 0
    
    def breakTo(self, ThrottlePersent = 0):
        self.ThrottlePersent = ThrottlePersent

class EngineMechanism:
    def __init__(self):
        self.throttleComponent = Throttle()
    
    def throttleRequest(self, in_ThrottleRate):
        self.throttleComponent.ThrottlePersent = in_ThrottleRate
    
    def returnEngineResult(self):
        return self.throttleComponent.ThrottlePersent






class SimCar:

    def __init__(self):
        self.car_SteeringMechanism = SteeringMechanism()
        self.car_BrakingMechanism = BrakingMechanism()
        self.car_EngineMechanism = EngineMechanism()
    
    def GetAlgInputAndWrite(self, wheel_to, brake_to, throttle_to):
        self.car_SteeringMechanism.steeringWheelRequest(wheel_to)
        self.car_BrakingMechanism.brakeRequest(brake_to)
        self.car_EngineMechanism.throttleRequest(throttle_to)
        
        self.__WriteResultToSM(self.__ValidateAlgInput())

    def __ValidateAlgInput(self):
        return [self.car_SteeringMechanism.returnSteeringResult(),
        self.car_BrakingMechanism.returnBrakingResult(),
        self.car_EngineMechanism.returnEngineResult()]

    def __WriteResultToSM(self, in_result):
        DataReceiver.writeAlgControllerSharedMemary(in_result)
    



