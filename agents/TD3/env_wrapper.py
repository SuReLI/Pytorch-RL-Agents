
import numpy as np
from gym.envs.classic_control import PendulumEnv
from gym.envs.box2d.lunar_lander import *

from utils import ReplayMemory

LunarLander.continuous = True


class PendulumWrapper(PendulumEnv):

    def reset(self):
        high = np.array([np.pi, 8])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()


class LunarWrapper(LunarLander):

    def forced_reset(self, x_test=VIEWPORT_W/SCALE/2, y_test=VIEWPORT_H/SCALE, angle_test=0.0,
                     init_force_x=0.0, init_force_y=0.0,
                     linearVelocity_test=None, angularVelocity_test=None,
                     seed=None):

        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,) )
        chunk_x  = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y  = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i],   smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append( [p1, p2, (p2[0],H), (p1[0],H)] )

        self.moon.color1 = (0.0,0.0,0.0)
        self.moon.color2 = (0.0,0.0,0.0)

        initial_x, initial_y = x_test, y_test
        self.lander = self.world.CreateDynamicBody(
            position = (initial_x, initial_y),
            angle=angle_test,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander.color1 = (0.5,0.4,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        self.lander.ApplyForceToCenter( (init_force_x, init_force_y), True)
        if linearVelocity_test is not None:
            self.lander.linearVelocity = linearVelocity_test
        if angularVelocity_test is not None:
            self.lander.angularVelocity = angularVelocity_test

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (initial_x - i*LEG_AWAY/SCALE, initial_y),
                angle = angle_test+(i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def reset(self):
        x = np.random.uniform(0, 20)
        y = np.random.uniform(5, 13)
        angle = np.random.uniform(0, 2*np.pi)
        linearVelocity = np.random.uniform(-5, 5, 2)
        angularVelocity = np.random.uniform(-5, 5)
        return self.forced_reset(x_test=x, y_test=y, angle_test=angle,
                                 linearVelocity_test=linearVelocity,
                                 angularVelocity_test=angularVelocity,
                                 seed=1234)

    def copy(self):
        x, y = self.lander.position
        angle = self.lander.angle
        linearVelocity = self.lander.linearVelocity.copy()
        angularVelocity = self.lander.angularVelocity
        new_env = LunarWrapper()
        new_env.forced_reset(x_test=x, y_test=y, angle_test=angle,
                             linearVelocity_test=linearVelocity,
                             angularVelocity_test=angularVelocity,
                             seed=1234)
        return new_env


def generate_memory(size, game='Pendulum'):

    if game.startswith('Pendulum'):
        env = PendulumWrapper()
    elif game.startswith('LunarLander'):
        env = LunarWrapper()

    memory = ReplayMemory(100000)

    for i in range(size):
        s = env.reset()
        a = env.action_space.sample()
        s_, r, d, _ = env.step(a)

        memory.push(s, a, r, s_, 1 - int(d))

    return memory
