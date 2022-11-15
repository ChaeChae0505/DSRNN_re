# real sensor version
import logging
from tkinter import INSIDE
from cv2 import sqrt
import gym
import numpy as np
import random
import pandas as pd
import os
import math


from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.orca import ORCA
from crowd_sim.envs.utils.state import *


from crowd_nav.policy.policy_factory import policy_factory

# The base class for our simulation environment

class CrowdSim(gym.Env):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.time_limit = None
        self.time_step = None
        self.robot = None # a Robot instance representing the robot
        self.humans = None # a list of Human instances, representing all humans in the environment
        self.global_time = None

        #  function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_dist_front = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None


        self.action_space=None
        self.observation_space=None

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        self.dummy_human = None
        self.dummy_robot = None

        #seed
        self.thisSeed=None # the seed will be set when the env is created

        #nenv
        self.nenv=None # the number of env will be set when the env is created.
        # Because the human crossing cases are controlled by random seed, we will calculate unique random seed for each
        # parallel env.

        self.phase=None # set the phase to be train, val or test
        self.test_case=None # the test case ID, which will be used to calculate a seed to generate a human crossing case

        # for render
        self.render_axis=None

        self.humans = []

        self.potential = None

    # configurate the environment with the given config
    def configure(self, config):
        self.config = config

        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.save_vis = config.sim.save_vis
        self.rStatic = config.sim.rStatic
        self.density = config.sim.density

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist_back
        self.discomfort_dist_front = config.reward.discomfort_dist_front
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.chae_dist = config.reward.chaeyoung_dist

        if self.config.humans.policy == 'orca' or self.config.humans.policy == 'social_force':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': self.config.env.val_size,
                              'test': self.config.env.test_size}
            self.circle_radius = config.sim.circle_radius
            self.human_num = config.sim.human_num
            self.group_human = config.sim.group_human

        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('Circle width: {}'.format(self.circle_radius))


        self.robot_fov = np.pi * config.robot.FOV
        self.human_fov = np.pi * config.humans.FOV
        logging.info('robot FOV %f', self.robot_fov)
        logging.info('humans FOV %f', self.human_fov)


        # set dummy human and dummy robot
        # dummy humans, used if any human is not in view of other agents
        self.dummy_human = Human(self.config, 'humans')
        # if a human is not in view, set its state to (px = 100, py = 100, vx = 0, vy = 0, theta = 0, radius = 0)
        self.dummy_human.set(7, 7, 7, 7, 0, 0, 0) # (7, 7, 7, 7, 0, 0, 0)
        self.dummy_human.time_step = config.env.time_step

        self.dummy_robot = Robot(self.config, 'robot')
        self.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        self.dummy_robot.time_step = config.env.time_step
        self.dummy_robot.kinematics = 'holonomic'
        self.dummy_robot.policy = ORCA(config)



        # configure noise in state
        self.add_noise = config.noise.add_noise
        if self.add_noise:
            self.noise_type = config.noise.type
            self.noise_magnitude = config.noise.magnitude


        # configure randomized goal changing of humans midway through episode
        self.random_goal_changing = config.humans.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.humans.goal_change_chance

        # configure randomized goal changing of humans after reaching their respective goals
        self.end_goal_changing = config.humans.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = config.humans.end_goal_change_chance

        # configure randomized radii changing when reaching goals
        self.random_radii = config.humans.random_radii

        # configure randomized v_pref changing when reaching goals
        self.random_v_pref = config.humans.random_v_pref

        # configure randomized goal changing of humans after reaching their respective goals
        self.random_unobservability = config.humans.random_unobservability
        if self.random_unobservability:
            self.unobservable_chance = config.humans.unobservable_chance


        self.last_human_states = np.zeros((self.human_num, 5))

        # configure randomized policy changing of humans every episode
        self.random_policy_changing = config.humans.random_policy_changing

        # set robot for this envs
        rob_RL = Robot(config, 'robot')
        self.set_robot(rob_RL)

        return


    def set_robot(self, robot):
        raise NotImplementedError


    # add all generated humans to the self.humans list
    def generate_random_human_position(self, human_num):
        """
        Generate human position: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        print()
        for i in range(human_num):
            self.humans.append(self.generate_circle_crossing_human())


    # generate and return a static human
    # position: (px, py) for fixed position, or None for random position
    def generate_circle_static_obstacle(self, position=None):
        # generate a human with radius = 0.3, v_pref = 1, visible = True, and policy = orca
        human = Human(self.config, 'humans')
        # For fixed position
        if position:
            px, py = position
        # For random position
        else:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                px_noise = (np.random.random() - 0.5) * v_pref
                py_noise = (np.random.random() - 0.5) * v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                                    norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

        # make it a static obstacle
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, px, py, 0, 0, 0, v_pref=0)
        return human


    # generate and return a moving human
    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            px_noise = (np.random.random() - 0.5) * v_pref
            py_noise = (np.random.random() - 0.5) * v_pref
            # px =  px_noise + np.random.randint(-0.5,0.5, size=1)
            # py = py_noise + np.random.randint(-0.5,5, size=1)
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False

            if self.group_human:
                collide = self.check_collision_group((px, py), human.radius)

            else:
                for i, agent in enumerate([self.robot] + self.humans):
                    # keep human at least 3 meters away from robot
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        min_dist = self.circle_radius / 2 # Todo: if circle_radius <= 4, it will get stuck here
                    else:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
            if not collide:
                break

        human.set(px, py, -px, -py, 0, 0, 0)
        return human


    # add noise according to env.config to observation
    def apply_noise(self, ob):
        if isinstance(ob[0], ObservableState):
            for i in range(len(ob)):
                if self.noise_type == 'uniform':
                    noise = np.random.uniform(-self.noise_magnitude, self.noise_magnitude, 5)
                elif self.noise_type == 'gaussian':
                    noise = np.random.normal(size=5)
                else:
                    print('noise type not defined')
                ob[i].px = ob[i].px + noise[0]
                ob[i].py = ob[i].px + noise[1]
                ob[i].vx = ob[i].px + noise[2]
                ob[i].vy = ob[i].px + noise[3]
                ob[i].radius = ob[i].px + noise[4]
            return ob
        else:
            if self.noise_type == 'uniform':
                noise = np.random.uniform(-self.noise_magnitude, self.noise_magnitude, len(ob))
            elif self.noise_type == 'gaussian':
                noise = np.random.normal(size = len(ob))
            else:
                print('noise type not defined')
                noise = [0] * len(ob)

            return ob + noise


    # update the robot belief of human states
    # if a human is visible, its state is updated to its current ground truth state
    # else we assume it keeps going in a straight line with last observed velocity
    def update_last_human_states(self, human_visibility, reset):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.human_num):
            if human_visibility[i]:
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS

            else:
                if reset:
                    humanS = np.array([15., 15., 0., 0., 0.3])
                    self.last_human_states[i, :] = humanS

                else:
                    px, py, vx, vy, r = self.last_human_states[i, :]
                    # Plan A: linear approximation of human's next position
                    px = px + vx * self.time_step
                    py = py + vy * self.time_step
                    self.last_human_states[i, :] = np.array([px, py, vx, vy, r])

                    # Plan B: assume the human doesn't move, use last observation
                    # self.last_human_states[i, :] = np.array([px, py, 0., 0., r])


    # return the ground truth locations of all humans
    def get_true_human_states(self):
        true_human_states = np.zeros((self.human_num, 2))
        for i in range(self.human_num):
            humanS = np.array(self.humans[i].get_observable_state_list())
            true_human_states[i, :] = humanS[:2]
        return true_human_states


    def randomize_human_policies(self):
        """
        Randomize the moving humans' policies to be either orca or social force
        """
        for human in self.humans:
            if not human.isObstacle:
                new_policy = random.choice(['orca','social_force'])
                new_policy = policy_factory[new_policy](self.config)
                human.set_policy(new_policy)


    # Generates group of circum_num humans in a circle formation at a random viable location
    def generate_circle_group_obstacle(self, circum_num):
        group_circumference = self.config.humans.radius * 2 * circum_num
        # print("group circum: ", group_circumference)
        group_radius = group_circumference / (2 * np.pi)
        # print("group radius: ", group_radius)
        while True:
            rand_cen_x = np.random.uniform(-3, 3)
            rand_cen_y = np.random.uniform(-3, 3)
            success = True
            for i, group in enumerate(self.circle_groups):
                # print(i)
                dist_between_groups = np.sqrt((rand_cen_x - group[1]) ** 2 + (rand_cen_y - group[2]) ** 2)
                sum_radius = group_radius + group[0] + 2 * self.config.humans.radius
                if dist_between_groups < sum_radius:
                    success = False
                    break
            if success:
                # print("------------\nsuccessfully found valid x: ", rand_cen_x, " y: ", rand_cen_y)
                break
        self.circle_groups.append((group_radius, rand_cen_x, rand_cen_y))

        # print("current groups:")
        # for i in self.circle_groups:
        #     print(i)

        arc = 2 * np.pi / circum_num
        for i in range(circum_num):
            angle = arc * i
            curr_x = rand_cen_x + group_radius * np.cos(angle)
            curr_y = rand_cen_y + group_radius * np.sin(angle)
            point = (curr_x, curr_y)
            # print("adding circle point: ", point)
            curr_human = self.generate_circle_static_obstacle(point)
            curr_human.isObstacle = True
            self.humans.append(curr_human)

        return

    # given an agent's xy location and radius, check whether it collides with all other humans
    # including the circle group and moving humans
    # return True if collision, False if no collision
    def check_collision_group(self, pos, radius):
        # check circle groups
        for r, x, y in self.circle_groups:
            if np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 2 * 0.5: # use 0.5 because it's the max radius of human
                return True

        # check moving humans
        for human in self.humans:
            if human.isObstacle:
                pass
            else:
                if np.linalg.norm([pos[0] - human.px, pos[1] - human.py]) <= human.radius + radius:
                    return True
        return False

    # check collision between robot goal position and circle groups
    def check_collision_group_goal(self, pos, radius):
        # print('check goal', len(self.circle_groups))
        collision = False
        # check circle groups
        for r, x, y in self.circle_groups:
            # print(np.linalg.norm([pos[0] - x, pos[1] - y]), r + radius + 4 * 0.5)
            if np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 4 * 0.5: # use 0.5 because it's the max radius of human
                collision = True
        return collision



    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        
        if human_num is None:
            human_num = self.human_num
        # for Group environment
        if self.group_human:
            # set the robot in a dummy far away location to avoid collision with humans
            self.robot.set(10, 10, 10, 10, 0, 0, np.pi / 2)

            # generate humans
            self.circle_groups = []
            humans_left = human_num
            
            # import random
            # number = []
            # for i in range(human_num):
            #     number.append(i)
            #     print(number)

            # random.shuffle(number)

            # # @LCY num 4 :  static test
            # while humans_left > 0:
            #     if humans_left == 1:
            #         print("che")
            #         circum_num = 1
            #         self.generate_circle_group_obstacle(circum_num)
            #         humans_left = 0
            #     else:
            #         print("yammy")
            #         self.generate_random_human_position(human_num=human_num-1)
            #         humans_left = 1

            while humans_left > 0:
                # print("****************\nhumans left: ", humans_left)
                if humans_left <= human_num - 5:#4: #4:# human_num-1: #4: num is static
                    if phase in ['train', 'val']:
                        self.generate_random_human_position(human_num=humans_left)
                    else:
                        self.generate_random_human_position(human_num=humans_left)
                    humans_left = 0
                else:
                    # if humans_left < 10:
                    #     max_rand = humans_left
                    # else:
                    #     max_rand = 10
                    # # print("randint from 4 to ", max_rand)
                    # circum_num = np.random.randint(4, max_rand)
                    circum_num = 1
                    # print("circum num: ", circum_num)
                    self.generate_circle_group_obstacle(circum_num)
                    humans_left -= circum_num
        # # for Group environment
        # if human_num is None:
        #     human_num = self.human_num
        # # for Group environment
        # if self.group_human:
        #     # set the robot in a dummy far away location to avoid collision with humans
        #     self.robot.set(10, 10, 10, 10, 0, 0, np.pi / 2)

        #     # generate humans
        #     self.circle_groups = []
        #     humans_left = human_num

        #     while humans_left > 0:
        #         # print("****************\nhumans left: ", humans_left)
        #         if humans_left <= 4:
        #             if phase in ['train', 'val']:
        #                 self.generate_random_human_position(human_num=humans_left)
        #             else:
        #                 self.generate_random_human_position(human_num=humans_left)
        #             humans_left = 0
        #         else:
        #             if humans_left < 10:
        #                 max_rand = humans_left
        #             else:
        #                 max_rand = 10
        #             # print("randint from 4 to ", max_rand)
        #             circum_num = np.random.randint(4, max_rand)
        #             # print("circum num: ", circum_num)
        #             self.generate_circle_group_obstacle(circum_num)
        #             humans_left -= circum_num

        #     # randomize starting position and goal position while keeping the distance of goal to be > 6
            # set the robot on a circle with radius 5.5 randomly
            rand_angle = np.random.uniform(0, np.pi * 2)
            # print('rand angle:', rand_angle)
            increment_angle = 0.0
            while True:
                px_r = np.cos(rand_angle + increment_angle) * 5.5
                py_r = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the initial px and py collides with any human
                collision = self.check_collision_group((px_r, py_r), self.robot.radius)
                # if the robot goal does not fall into any human groups, the goal is okay, otherwise keep generating the goal
                if not collision:
                    #print('initial pos angle:', rand_angle+increment_angle)
                    break
                increment_angle = increment_angle + 0.2

            increment_angle = increment_angle + np.pi # start at opposite side of the circle
            while True:
                gx = np.cos(rand_angle + increment_angle) * 5.5
                gy = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the goal is inside the human groups
                # check whether the initial px and py collides with any human
                collision = self.check_collision_group_goal((gx, gy), self.robot.radius)
                # if the robot goal does not fall into any human groups, the goal is okay, otherwise keep generating the goal
                if not collision:
                    # print('goal pos angle:', rand_angle + increment_angle)
                    break
                increment_angle = increment_angle + 0.2

            self.robot.set(px_r, py_r, gx, gy, 0, 0, np.pi / 2)

        # for FoV environment
        else:
            if self.robot.kinematics == 'unicycle':
                angle = np.random.uniform(0, np.pi * 2)
                px = self.circle_radius * np.cos(angle)
                py = self.circle_radius * np.sin(angle)
                while True:
                    gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 2)
                    if np.linalg.norm([px - gx, py - gy]) >= 6:  # 1 was 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2*np.pi)) # randomize init orientation

            # randomize starting position and goal position
            else:
                while True:
                    px, py, gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 4)
                    if np.linalg.norm([px - gx, py - gy]) >= 6:
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.pi/2)


            # generate humans
            self.generate_random_human_position(human_num=human_num)


    # reset function
    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0

        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)

        self.generate_robot_humans(phase)


        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get current observation
        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        return ob


    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def update_human_goals_randomly(self):
        # Update humans' goals randomly
        for human in self.humans:
            if human.isObstacle or human.v_pref == 0:
                continue
            if np.random.random() <= self.goal_change_chance:
                if not self.group_human: # to improve the runtime
                    humans_copy = []
                    for h in self.humans:
                        if h != human:
                            humans_copy.append(h)


                # Produce valid goal for human in case of circle setting
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False

                    if self.group_human:
                        collide = self.check_collision_group((gx, gy), human.radius)
                    else:
                        for agent in [self.robot] + humans_copy:
                            min_dist = human.radius + agent.radius + self.discomfort_dist
                            if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                    norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                                collide = True
                                break
                    if not collide:
                        break

                # Give human new goal
                human.gx = gx
                human.gy = gy
        return

    # Update the specified human's end goals in the environment randomly
    def update_human_goal(self, human):
        # Update human's goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            if not self.group_human:
                humans_copy = []
                for h in self.humans:
                    if h != human:
                        humans_copy.append(h)

            # Update human's radius now that it's reached goal
            if self.random_radii:
                human.radius += np.random.uniform(-0.1, 0.1)

            # Update human's v_pref now that it's reached goal
            if self.random_v_pref:
                human.v_pref += np.random.uniform(-0.1, 0.1)

            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                if self.group_human:
                    collide = self.check_collision_group((gx, gy), human.radius)
                else:
                    for agent in [self.robot] + humans_copy:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                            collide = True
                            break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy
        return

    # Caculate whether agent2 is in agent1's FOV
    # Not the same as whether agent1 is in agent2's FOV!!!!
    # arguments:
    # state1, state2: can be agent instance OR state instance
    # robot1: is True if state1 is robot, else is False
    # return value:
    # return True if state2 is visible to state1, else return False
#     def detect_visible(self, state1, state2, robot1 = False, custom_fov=None):
#         if self.robot.kinematics == 'holonomic':
#             real_theta = np.arctan2(state1.vy, state1.vx)
#         else:
#             real_theta = state1.theta
#         # angle of center line of FOV of agent1
#         v_fov = [np.cos(real_theta), np.sin(real_theta)]

#         # angle between agent1 and agent2
#         v_12 = [state2.px - state1.px, state2.py - state1.py]
#         # angle between center of FOV and agent 2
#         v_fov = v_fov / np.linalg.norm(v_fov)
#         distance = np.linalg.norm(v_12)
#         v_12 = v_12 / np.linalg.norm(v_12)
        

#         offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        
#         if custom_fov:
#             fov = custom_fov
#         else:
#             if robot1:
#                 fov = self.robot_fov
#             else:
#                 fov = self.human_fov
#         # print(offset)
#         if np.abs(offset) <= fov / 2:
#             self.angle_mask = round(offset,1)
#             self.dis_mask = distance
#             return True 
#         else:
#             self.angle_mask = round(offset,1)
#             self.dis_mask = distance
#             return False


#     # for robot:
#     # return only visible humans to robot and number of visible humans and visible humans' ids (0 to 4)

#     # def get_num_human_in_fov(self):
#     #     human_ids = []
#     #     humans_in_view = []
#     #     num_humans_in_view = 0

#     #     for i in range(self.human_num):
#     #         visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
#     #         if visible:
#     #             humans_in_view.append(self.humans[i])
#     #             num_humans_in_view = num_humans_in_view + 1
#     #             human_ids.append(True)
#     #         else:
#     #             human_ids.append(False)

#     #     return humans_in_view, num_humans_in_view, human_ids

#     def get_num_human_in_fov(self):
#         human_ids = []
#         humans_in_view = []
#         num_humans_in_view = 0
#         num_human = 0 
#         list = []
#         human_in = []


#         for i in range(self.human_num):
#             visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
#             angle, dis = self.angle_mask, self.dis_mask
#             if visible:
#                 humans_in_view.append(self.humans[i])
#                 num_humans_in_view = num_humans_in_view + 1
#                 human_ids.append(True)
#                 list.append([num_humans_in_view, visible, angle, dis])
#             else:
#                 human_ids.append(False)
#         sort_list= sorted(list, key=lambda list:-list[3])
#         sort_list = np.array(sort_list)
#         print(angle)
#         check = []

#         for num in range(num_humans_in_view):
#             # print(sort_list.shape)
            
#             if sort_list[num, 2] not in check:
#                 # print("print", sort_list[num,2], num)
#                 check.append(sort_list[num, 2])
#                 num_human = num_human + 1
#                 human_in.append(humans_in_view[num])

#             else:
#                 human_ids[num] = False

#         print(human_in, num_human, human_ids)     

#         return human_in, num_human, human_ids

	def detect_visible(self, state1, state2, robot1 = False, custom_fov=None, is_occlusion_considered=False):
	    if self.robot.kinematics == 'holonomic':
		real_theta = np.arctan2(state1.vy, state1.vx)
	    else:
		real_theta = state1.theta
	    # angle of center line of FOV of agent1
	    v_fov = [np.cos(real_theta), np.sin(real_theta)]

	    # angle between agent1 and agent2
	    v_12 = [state2.px - state1.px, state2.py - state1.py]
	    # angle between center of FOV and agent 2
	    v_fov = v_fov / np.linalg.norm(v_fov)
	    distance = np.linalg.norm(v_12)
	    v_12 = v_12 / np.linalg.norm(v_12)

	    offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))

	    if custom_fov:
		fov = custom_fov
	    else:
		if robot1:
		    fov = self.robot_fov
		else:
		    fov = self.human_fov

	    # a variable to check if the state2 is in fov of state1
	    is_in_fov = False
	    if np.abs(offset) <= fov / 2:
		is_in_fov = True
	    else:
		is_in_fov = False

	    if is_occlusion_considered:
		return is_in_fov, v_12, distance
	    else:
		return is_in_fov

	def get_num_human_in_fov(self):
	    human_ids = []
	    humans_in_view = []
	    num_humans_in_view = 0
	    visibility_info_arr = np.zeros([self.human_num, 4])
	    occulsion_angle_threshold = 0.01

	    for i in range(self.human_num):
		visible, unit_vector, distance = self.detect_visible(self.robot, self.humans[i], robot1=True, is_occlusion_considered=True)
		# store visibility information
		visibility_info_arr[i,0] = visible
		visibility_info_arr[i,1:3] = unit_vector
		visibility_info_arr[i,3] = distance

	    # get indices of visible humans
	    visible_human_idx = np.where(visibility_info_arr[:,0])[0]
	    info_visible = visibility_info_arr[visible_human_idx]
	    # compute angle difference between humans
	    angle_diff = info_visible[:,1:3].dot(info_visible[:,1:3].T) - np.eye(len(info_visible))
	    angle_diff = 1 - angle_diff
	    angle_occlusion_mask = angle_diff < occulsion_angle_threshold
	    # compute distance difference between humans
	    dist_diff = info_visible[:,3][:,None] - info_visible[:,3][None,:]
	    dist_occlusion_mask = dist_diff > 0
	    # compute occulusion mask
	    occlusion_mask = np.any(angle_occlusion_mask*dist_occlusion_mask, axis=1)
	    info_visible[:,0] = info_visible[:,0]*(1 - occlusion_mask) + False*occlusion_mask
	    # update visbility
	    visibility_info_arr[visible_human_idx, 0] = True*info_visible[:,0]

	    # arrange info
	    human_ids = visibility_info_arr[:,0].tolist()
	    humans_in_view = [self.humans[i] for idx in range(self.human_num) if human_ids[idx]]
	    num_humans_in_view = len(humans_in_view)

	    return humans_in_view, num_humans_in_view, human_ids


    # def detect_visible(self, state1, state2, robot1 = False, custom_fov=None):
    #     if self.robot.kinematics == 'holonomic':
    #         real_theta = np.arctan2(state1.vy, state1.vx)
    #     else:
    #         real_theta = state1.theta
    #     # angle of center line of FOV of agent1
    #     v_fov = [np.cos(real_theta), np.sin(real_theta)]

    #     # angle between agent1 and agent2
    #     v_12 = [state2.px - state1.px, state2.py - state1.py]
    #     # angle between center of FOV and agent 2

    #     v_fov = v_fov / np.linalg.norm(v_fov)
    #     v_12 = v_12 / np.linalg.norm(v_12)

    #     offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
    #     if custom_fov:
    #         fov = custom_fov
    #     else:
    #         if robot1:
    #             fov = self.robot_fov
    #         else:
    #             fov = self.human_fov

    #     if np.abs(offset) <= fov / 2:
    #         return True
    #     else:
    #         return False

    # def cal_incli(self, x1, y1, x2, y2):
    #     a = (y2 - y1)/(x2 - x1)
    #     return a

    # # for robot:
    # # return only visible humans to robot and number of visible humans and visible humans' ids (0 to 4)
    # def get_num_human_in_fov(self):
    #     human_ids = []
    #     humans_in_view = []
    #     num_humans_in_view = 0

    #     human_sensor_invisible = []
    #     human_robot_incli = {}

    #     for i in range(self.human_num):
    #         visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
    #         if visible:
    #             humans_in_view.append(self.humans[i])
    #             print(len(humans_in_view))
    #             num_humans_in_view = num_humans_in_view + 1
    #             human_ids.append(True)
    #             # robot과 human 계산 해서 기울기일치하면 제외하기 
    #             # print(self.robot.px, self.robot.py, self.humans[i].px, self.humans[i].py)
    #             incli = self.cal_incli(self.robot.px, self.robot.py, self.humans[i].px, self.humans[i].py) # 기울기 calculate
    #             human_robot_incli[i]=1#round(incli,1)

    #         else:
    #             human_ids.append(False)
    #             incli = self.cal_incli(self.robot.px, self.robot.py, self.humans[i].px, self.humans[i].py) # 기울기 calculate
    #             human_robot_incli[i]=1#round(incli,1)
        
    #     key_value = human_robot_incli.items()
    #     key_list = list(key_value)
    #     # print(human_robot_incli, list(key_value), key_list[0][1] )
    #     human_ids_sensor = []
    #     x = []
    #     for num, i in enumerate(human_robot_incli.values()):
    #         if i not in x:
    #             x.append(i)
    #         else:
    #             human_ids_sensor.append(num)
    #     # print(human_ids)
    #     human_ids = np.array(human_ids)
    #     np.put(human_ids, human_ids_sensor, False)
    #     human_ids.tolist()
    #     # print(human_ids)
			
    #     return humans_in_view, num_humans_in_view, human_ids


    # convert an np array with length = 34 to a JointState object
    def array_to_jointstate(self, obs_list):
        fullstate = FullState(obs_list[0], obs_list[1], obs_list[2], obs_list[3],
                              obs_list[4],
                              obs_list[5], obs_list[6], obs_list[7], obs_list[8])

        observable_states = []
        for k in range(self.human_num):
            idx = 9 + k * 5
            observable_states.append(
                ObservableState(obs_list[idx], obs_list[idx + 1], obs_list[idx + 2],
                                obs_list[idx + 3], obs_list[idx + 4]))
        state = JointState(fullstate, observable_states)
        return state


    def last_human_states_obj(self):
        '''
        convert self.last_human_states to a list of observable state objects for old algorithms to use
        '''
        humans = []
        for i in range(self.human_num):
            h = ObservableState(*self.last_human_states[i])
            humans.append(h)
        return humans

    # def robot_inside(self, rx, ry, rad, hx, hy):
    #     x = (hx - rx)**2
    #     y = (hy - ry)**2
    #     if x + y <=rad**2:
    #         return True
    #     else:
    #         return False

    # def human_safe_zone(self, hx, hy, hvx, hvy, density, rx, ry):
    #     result_true = 1
    #     result_false = 0
    #     PI = math.pi
    #     v = (hvx**2 + hvy **2) ** (1/2)
    #     R = density*v + self.rStatic # input config
    #     v_theta = math.atan2(hvy, hvx) # human theta
    #     print(R)
    #     theta = (11*PI / 6) * math.exp(-1.4*v) + (PI / 6)
        
    #     dx = hx - rx
    #     dy = hy - ry
    #     d_c = (dx**2 + dy**2) ** (1/2)
        
    #     alpha = math.atan2(dy,dx) # robot, human theta
    #     # if alpha > v_theta - (theta/2) and alpha < v_theta + (theta/2):
    #         # print(d_c, v, R, alpha*180/PI, v_theta*180/PI, theta*180/PI, dx, dy)
            
    #     # print(v, R, alpha*180/PI, v_theta*180/PI, theta*180/PI, dx, dy, d_c)
    #     if dx**2 + dy ** 2 <= R**2:
    #         if alpha > v_theta - (theta/2) and alpha < v_theta + (theta/2):
    #             # print(v, R, alpha*180/PI, v_theta*180/PI, theta*180/PI, dx, dy, d_c)

    #             return result_true, d_c, R
    #         else:
    #             # print("case 1")
    #             return result_false, d_c, R
    #     else:
    #         # print("case2")
    #         return result_false, d_c, R

    # find R(s, a), done or not, and episode information
    def calc_reward(self, action):
        # collision detection
        dmin = float('inf')
        danger_dists = []
        collision = False

        robot_range_radius = 6
        inside = []
        dis_penalty = []

        # # 1) How many people inside the robot range
        # for i, human in enumerate(self.humans):

        #     r_sensor_range = self.robot_inside(self.robot.px, self.robot.py, robot_range_radius, human.px, human.py)
        #     if r_sensor_range:
        #         inside.append(i)
        #         print(inside, len(inside)) #len is inside total human num
        
         

     
        # for i in inside:
        #     check_True , d_c, R = self.human_safe_zone(self.humans[i].px, self.humans[i].py, 
        #     self.humans[i].vx, self.humans[i].vy, self.density, self.robot.px, self.robot.py)
        #     # print(i, check_True, d_c, R)
        # # print(math.atan2(self.humans[3].vy, self.humans[3].vx)*180/math.pi)

        for i, human in enumerate(self.humans):
            check = self.humans[i].get_observable_state() # @LCY

            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
        
            speed = (human.vx ** 2 + human.vy ** 2) ** (1/2)
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius
            if closest_dist < self.chae_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist
                num = i 
                #@LCY
                checking = check.check
                # print(checking, speed)
                #@LCY


        # check if reaching the goal
        reaching_goal = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()
        elif dmin < self.chae_dist: # When testing the result you chaning config.chae_dist = 0.25
        # elif dmin < speed * 0.55: 
            reward = (dmin - self.chae_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(dmin)
        
        # elif dmin < self.chae_dist: 
        #     reward = (dmin - self.chae_dist) * self.discomfort_penalty_factor * self.time_step
        #     done = False
        #     episode_info = Danger(dmin)
   
        else:
            # potential reward
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = 2 * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()

        

        if self.robot.kinematics == 'unicycle':
            # add a rotational penalty
            # if action.r is w, factor = -0.02 if w in [-1.5, 1.5], factor = -0.045 if w in [-1, 1];
            # if action.r is delta theta, factor = -2 if r in [-0.15, 0.15], factor = -4.5 if r in [-0.1, 0.1]
            r_spin = -2 * action.r**2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.
            # print(reward, r_spin, r_back)
            reward = reward + r_spin + r_back

        return reward, done, episode_info




    # # find R(s, a), done or not, and episode information
    # def calc_reward(self, action):
    #     # collision detection
    #     dmin = float('inf')

    #     danger_dists = []
    #     collision = False

    #     for i, human in enumerate(self.humans):
    #         # @LCY
    #         check = self.humans[i].get_observable_state()
    #         # print("check" , check.check)
    #         # @LCY
    #         dx = human.px - self.robot.px
    #         dy = human.py - self.robot.py
        
    #         speed = (human.vx ** 2 + human.vy ** 2) ** (1/2)
    #         # print(speed)
    #         closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius
    #         # print((dx ** 2 + dy ** 2),(dx ** 2 + dy ** 2) ** (1 / 2),human.radius, self.robot.radius )
    #         if closest_dist < self.discomfort_dist:
    #             danger_dists.append(closest_dist)
    #         if closest_dist < 0:
    #             collision = True
    #             # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
    #             break
    #         elif closest_dist < dmin:
    #             dmin = closest_dist
    #             num = i 
    #             #@LCY
    #             checking = check.check
    #             # print(checking, speed)
    #             #@LCY


    #     # check if reaching the goal
    #     reaching_goal = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius

    #     if self.global_time >= self.time_limit - 1:
    #         reward = 0
    #         done = True
    #         episode_info = Timeout()
    #     elif collision:
    #         reward = self.collision_penalty
    #         done = True
    #         episode_info = Collision()
    #     elif reaching_goal:
    #         reward = self.success_reward
    #         done = True
    #         episode_info = ReachGoal()
    #     #@LCY
    #     elif dmin < self.discomfort_dist: # Method 1
    #     # # elif (num != 1 and dmin < self.discomfort_dist) or (num == 1 and dmin < self.discomfort_dist - 0.1): # Method 2
    #     # # elif dmin < self.discomfort_dist - speed * 0.1: # Method 3 (False)
    #     # # elif dmin < (self.discomfort_dist - 0.1) + speed * 0.1: # Method 3 (Right) 0.15 ~ 0.25

    #     # elif dmin < self.discomfort_dist - 0.1: # Method 4

    #         # only penalize agent for getting too close if it's visible
    #         # adjust the reward based on FPS
    #         # print(dmin, self.discomfort_dist) dmin < 0.25 or dmin < 0.15
    #         # print("self.discomfort_dist - 0.1", )
    #         # @LCY save distance
    #         # print("reward", self.discomfort_dist - speed*0.1)
    #         # print("original", self.discomfort_dist, self.discomfort_dist - 0.1)
    #         # df_check = pd.DataFrame({'velocity':[checking],'dmin':[dmin]})
    #         # if os.path.exists(os.path.join('./distance.csv')):
    #         #     df_check.to_csv(os.path.join('./distance.csv'), mode='a', header=False, index=False)
    #         # else:
    #         #     df_check.to_csv(os.path.join('./distance.csv'), mode='w', header=True, index=False)

    #         discomfort_dist = self.discomfort_dist
    #         # if checking == 1:
    #         #     discomfort_dist = self.discomfort_dist
    #         # else:
    #         #     discomfort_dist = self.discomfort_dist + 0.1 
    #         reward = (dmin - discomfort_dist) * self.discomfort_penalty_factor * self.time_step

    #         done = False
    #         episode_info = Danger(dmin)
    #     # #@LCY
    #     # elif dmin < self.discomfort_dist:
    #     #     # only penalize agent for getting too close if it's visible
    #     #     # adjust the reward based on FPS
    #     #     # print(dmin)
    #     #     df_check = pd.DataFrame({'velocity':[checking],'dmin':[dmin]})
    #     #     if os.path.exists(os.path.join('./distance.csv')):
    #     #         df_check.to_csv(os.path.join('./distance.csv'), mode='a', header=False, index=False)
    #     #     else:
    #     #         df_check.to_csv(os.path.join('./distance.csv'), mode='w', header=True, index=False)

    #     #     discomfort_dist = self.discomfort_dist
    #     #     reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
    #     #     done = False
    #     #     episode_info = Danger(dmin)
    #     else:
    #         # potential reward
    #         potential_cur = np.linalg.norm(
    #             np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
    #         reward = 2 * (-abs(potential_cur) - self.potential)
    #         self.potential = -abs(potential_cur)
    #         #@LCY 
    #         # if dmin < self.discomfort_dist:
    #         #     discomfort_dist = self.discomfort_dist
    #         #     reward_r = (dmin - discomfort_dist) * self.discomfort_penalty_factor * self.time_step
    #         #     done = False
    #         #     episode_info = Danger(dmin)
    #         #     reward = reward + reward_r
    #         done = False
    #         episode_info = Nothing()

        

    #     # if the robot is near collision/arrival, it should be able to turn a large angle
    #     if self.robot.kinematics == 'unicycle':
    #         # add a rotational penalty
    #         # if action.r is w, factor = -0.02 if w in [-1.5, 1.5], factor = -0.045 if w in [-1, 1];
    #         # if action.r is delta theta, factor = -2 if r in [-0.15, 0.15], factor = -4.5 if r in [-0.1, 0.1]
    #         r_spin = -2 * action.r**2

    #         # add a penalty for going backwards
    #         if action.v < 0:
    #             r_back = -2 * abs(action.v)
    #         else:
    #             r_back = 0.
    #         # print(reward, r_spin, r_back)
    #         reward = reward + r_spin + r_back

    #     return reward, done, episode_info






    # compute the observation as a flattened array
    def generate_ob(self, reset):
        visible_human_states, num_visible_humans, self.human_visibility = self.get_num_human_in_fov()
        self.update_last_human_states(self.human_visibility, reset=reset)
        if self.robot.policy.name in ['lstm_ppo', 'srnn']:
            ob = [num_visible_humans]
            # append robot's state
            robotS = np.array(self.robot.get_full_state_list())
            ob.extend(list(robotS))

            ob.extend(list(np.ravel(self.last_human_states)))
            ob = np.array(ob)

        else: # for orca and sf
            ob = self.last_human_states_obj()

        if self.add_noise:
            ob = self.apply_noise(ob)

        return ob

    # get the actions for all humans at the current timestep
    def get_human_actions(self):
        # step all humans
        human_actions = []  # a list of all humans' actions
        for i, human in enumerate(self.humans):
            # observation for humans is always coordinates
            ob = []

            for other_human in self.humans:
                if other_human != human:
                    # Chance for one human to be blind to some other humans
                    if self.random_unobservability and i == 0:
                        if np.random.random() <= self.unobservable_chance or not self.detect_visible(human,
                                                                                                     other_human):
                            ob.append(self.dummy_human.get_observable_state())
                        else:
                            ob.append(other_human.get_observable_state())
                    # Else detectable humans are always observable to each other
                    elif self.detect_visible(human, other_human):
                        ob.append(other_human.get_observable_state())
                    else:
                        ob.append(self.dummy_human.get_observable_state())

            if self.robot.visible:
                # Chance for one human to be blind to robot
                if self.random_unobservability and i == 0:
                    if np.random.random() <= self.unobservable_chance or not self.detect_visible(human, self.robot):
                        ob += [self.dummy_robot.get_observable_state()]
                    else:
                        ob += [self.robot.get_observable_state()]
                # Else human will always see visible robots
                elif self.detect_visible(human, self.robot):
                    ob += [self.robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]

            human_actions.append(human.act(ob))
        return human_actions


    # step function
    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        # clip the action to obey robot's constraint
        action = self.robot.policy.clip_action(action, self.robot.v_pref)

        # step all humans
        human_actions = self.get_human_actions()


        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)


        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step # max episode length=time_limit/time_step

        ##### compute_ob goes here!!!!!
        ob = self.generate_ob(reset=False)


        if self.robot.policy.name in ['srnn']:
            info={'info':episode_info}
        else: # for orca and sf
            info=episode_info

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()


        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if not human.isObstacle and human.v_pref != 0 and norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)

        return ob, reward, done, info


    # render function
    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint



        ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)


        # draw FOV for the robot
        # add robot FOV
        if self.robot_fov < np.pi * 2:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=True) for human in self.humans]
       

        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])
            # print(self.human_visibility)
            # green: visible; red: invisible
            if self.human_visibility[i]:
                human_circles[i].set_color(c='g')
            else:
                human_circles[i].set_color(c='r')
            plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(i), color='black', fontsize=12)

        plt.pause(0.1)
        # for item in artists:
        #     item.remove() # there should be a better way to do this. For example,
        #     # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)

        if self.save_vis:
            return artists
