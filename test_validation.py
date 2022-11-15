# folder에 평가하고 싶은 pt 파일 추가하기
# evluation file에서 평가 데이터 그냥 저장 할 수 있도록 csv

import os
import subprocess


result = []
path = '/home/a307/collision/220915/Validation_Test/data/5s5d_time/checkpoints'
file_list = os.listdir(path)

file_list_print = [file for file in file_list if file.endswith('.pt')]
file_list_print.sort()
# print(len(file_list_print), file_list_print)

for file in file_list_print:
    call_name = 'python test.py'+' '+'--test_model'+' '+file
    print(call_name)
    subprocess.run(call_name, shell=True)
    result.append(file)
print(result)


# evaluate.py

import numpy as np
from sentry_sdk import last_event_id
import torch

from crowd_sim.envs.utils.info import *
from pytorchBaselines.a2c_ppo_acktr import utils

import pandas as pd
import os
import csv

def evaluate(actor_critic, ob_rms, eval_envs, num_processes, device, config, logging, visualize=False, save_vis=False,
             recurrent_type='GRU'):

    test_size = config.env.test_size

    if ob_rms:
        vec_norm = utils.get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    if recurrent_type == 'LSTM':
        rnn_factor = 2
    else:
        rnn_factor = 1


    eval_recurrent_hidden_states = {}

    node_num = 1
    edge_num = actor_critic.base.human_num + 1
    eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, config.SRNN.human_node_rnn_size * rnn_factor,
                                                                 device=device)

    eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                       config.SRNN.human_human_edge_rnn_size*rnn_factor,
                                                                       device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []
    path_success = []
    chc_total = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0.
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    gamma = 0.99
    baseEnv = eval_envs.venv.envs[0].env

    if save_vis:
        artists = []

    obs = eval_envs.reset()
    for k in range(test_size):
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0

        global_time = 0.0
        path = 0.0
        chc = 0.0

        last_pos = obs['robot_node'][0, 0, 0:2].cpu().numpy()  # robot px, py
        last_angle = np.arctan2(obs['temporal_edges'][0, 0, 1].cpu().numpy(), obs['temporal_edges'][0, 0, 0].cpu().numpy())  # robot theta
        
        #@LCY
        if os.path.exists('x_y_robot.csv'):
            f = open('x_y_robot.csv','a', newline='')
            wr = csv.writer(f)
            wr.writerow([k])
            ff = open('result.csv','a', newline='')
            wrr = csv.writer(ff)
            wrr.writerow([k])
        

        while not done:
            stepCounter = stepCounter + 1
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
            if not done:
                global_time = baseEnv.global_time
            if visualize:
                art_tmp = eval_envs.render()
                if save_vis:
                    artists.append(art_tmp)
            # print("actor_critic.act", action)
            # print(last_pos) # robot action
            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            path = path + np.linalg.norm(np.array([last_pos[0] - obs['robot_node'][0, 0, 0].cpu().numpy(),
                                                   last_pos[1] - obs['robot_node'][0, 0, 1].cpu().numpy()]))
            cur_angle = np.arctan2(obs['temporal_edges'][0, 0, 1].cpu().numpy(), obs['temporal_edges'][0, 0, 0].cpu().numpy())
            chc = chc +  abs(cur_angle - last_angle)

            last_pos = obs['robot_node'][0, 0, 0:2].cpu().numpy()  # robot px, py
            
            #@LCY 
            robot_x = last_pos[0]
            robot_y = last_pos[1]
            # print("Robot x, y", robot_x, robot_y)

            df = pd.DataFrame([{'robot_x':last_pos[0],'robot_y':last_pos[1]}])

            if os.path.exists('x_y_robot.csv'):
                df.to_csv('x_y_robot.csv', mode='a', header=False, index=False)
            else:
                df.to_csv('x_y_robot.csv', mode='w', header=True, index=False)


            #@@
            last_angle = cur_angle
            rewards.append(rew)


            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

                # #@LCY
                # print("min", infos[0]['info'].min_dist)
                # df2 = pd.DataFrame([{'min_dist':infos[0]['info'].min_dist}])
                # df2.to_csv('x_y_robot.csv', mode='a', header=True, index=False)
            
            episode_rew += rew[0]



            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])


        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        
        path_lengths.append(path)
        chc_total.append(chc)

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            path_success.append(path)
            print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            print('Time out')
        else:
            raise ValueError('Invalid end signal from environment')
        df = pd.DataFrame({'info': [infos[0]['info']],'step' : stepCounter, 'reward' : [episode_rew[0]], 'path': [path], 'time': [global_time]})
        # if os.path.exists(os.path.join(config.training.output_dir, 'path.csv')):
        #     df.to_csv(os.path.join(config.training.output_dir, 'path.csv'), mode='a', header=False, index=False)
        # else:
        #     df.to_csv(os.path.join(config.training.output_dir, 'path.csv'), mode='w', header=True, index=False)



        cumulative_rewards.append(sum([pow(gamma, t * baseEnv.robot.time_step * baseEnv.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]))


    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else baseEnv.time_limit  # baseEnv.env.time_limit

    extra_info = ''
    phase = 'test'
    logging.info(
        '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, total reward: {:.4f}'.
            format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, avg_nav_time,
                   np.average(cumulative_rewards)))
    if phase in ['val', 'test']:
        total_time = sum(success_times + collision_times + timeout_times)
        if min_dist:
            avg_min_dist = np.average(min_dist)
        else:
            avg_min_dist = float("nan")
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                     too_close * baseEnv.robot.time_step / total_time, avg_min_dist)
        
    logging.info(
        '{:<5} {}has average path length: {:.2f}, {:.2f},  CHC: {:.2f}'.
            format(phase.upper(), extra_info, sum(path_lengths) / test_size, sum(path_success)/len(path_success), sum(chc_total) / test_size)) #len(path_success)
    # print(len(success_times), len(path_lengths), test_size) #success time !=testsize
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    print('Path length var : {:.2f}, std : {:.2f}\n'.format(np.var(path_success), np.std(path_success)))
    print('Navigation time var : {:.2f}, std : {:.2f}'.format(np.var(success_times), np.std(success_times)))

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    dff = pd.DataFrame({'Success_reward': [config.reward.success_reward], 'Collision': [config.reward.collision_penalty], 'Human_group': [config.sim.group_human] , 'Success-rate':[success_rate],'Collision-rate':[collision_rate],'Timeout-rate':[timeout_rate] , 
                        'Frequency':[too_close * baseEnv.robot.time_step / total_time],'avg_min_dist' : [avg_min_dist], 
                        'avg_nav_time':[avg_nav_time],'path_time':[sum(path_success)/len(path_success)] })
    if os.path.exists(os.path.join(os.getcwd(), '5d_1_3.csv')):
        dff.to_csv(os.path.join(os.getcwd(), '5d_1_3.csv'), mode='a', header=True, index=False)
    else:
        dff.to_csv(os.path.join(os.getcwd(), '5d_1_3.csv'), mode='w', header=True, index=False)
    
    if save_vis:
        return artists

def method2(methodToRun):
    x, y = methodToRun()
    return x, y
