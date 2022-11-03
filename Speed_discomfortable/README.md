
```python
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
        
        tmp_danzer_dist = []
        tmp_next_dist = []
        next_collision_reward = 0.
        next_closest_reward = 0.
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
            for j in range(1,4):
                hx = human.px + human.vx * self.time_step*j
                hy = human.py + human.vy * self.time_step*j
                d_x = hx - self.robot.px
                d_y = hy - self.robot.py
                next_closest_dist = (d_x ** 2 + d_y ** 2) ** (1 / 2) - human.radius - self.robot.radius
                if next_closest_dist < 0:
                    next_collision_reward = next_collision_reward - (math.exp(1.4*-j))
                    # tmp_danzer_dist.append([i, j, next_closest_dist])
                elif next_closest_dist < self.chae_dist:
                    next_closest_reward = next_closest_reward - (math.exp(1.8*-j))
                    # tmp_next_dist.append([i, j, next_closest_dist])
                


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
        #elif dmin < self.chae_dist: # When testing the result you chaning config.chae_dist = 0.25
        elif dmin < self.chae_dist: 
            reward2 = (dmin - self.chae_dist) * self.discomfort_penalty_factor * self.time_step
            reward = reward2 + next_collision_reward + next_closest_reward
            print(reward, reward2)
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
            # reward = reward2 + next_collision_reward + next_closest_reward
            # print("nothing", reward, reward2)

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
```
