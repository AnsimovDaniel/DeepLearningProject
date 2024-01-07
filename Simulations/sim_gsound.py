import numpy as np
import pygsound as ps
import json
import argparse
from wavefile import WaveWriter
import re
import os
import pandas as pd

CONFIG_NAME = 'sim_config.json'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    type = str,
    help = 'path to input folder'
)

parser.add_argument(
    '--nthreads',
    default = 0,
    type = int,
    help = 'number of threads to use'
)

def add_stat(stats, mic_loc, source_loc, save_path):
    stats = pd.concat([stats, pd.DataFrame({'mic x':mic_loc[0], 'mic y':mic_loc[1], 'mic z':mic_loc[2], 'source x':source_loc[0], 'source y':source_loc[1], 'source z':source_loc[2]}, index=[save_path])])
    return stats



args = parser.parse_args()
folder_path = args.input
config_path = os.path.join(folder_path, CONFIG_NAME)
save_folder = os.path.join(folder_path, 'geo')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

ctx = ps.Context()
ctx.diffuse_count = 20000
ctx.specular_count = 2000
ctx.specular_depth = 10
ctx.channel_type = ps.ChannelLayoutType.mono
ctx.sample_rate = 48000
if args.nthreads > 0:
    ctx.threads_count = args.nthreads
ctx.normalize = False

stats = pd.DataFrame(columns=['mic x', 'mic y', 'mic z', 'source x', 'source y', 'source z'], index=pd.Index([], name='path'))


with open(config_path, 'r') as f:
    data = json.load(f)
    if os.path.isabs(data['obj_path']):
        obj_path = data['obj_path']
    else:
        obj_path = os.path.join(os.path.dirname(config_path), data['obj_path'])
    mesh = ps.loadobj(obj_path)
    scene = ps.Scene()
    scene.setMesh(mesh)

    for i in range(0,15):
        print(i)
        for j in range(0,15):
            for k in range(0, 20):
                src_locs = []
                lis_locs = []
                src_idx = []
                lis_idx = []
                for source in data['sources']:
                    src_idx.append(int(re.findall(r'\d+', source['name'])[0]))
                    src_locs.append([source['xyz'][0] + i * 0.18, source['xyz'][1] + j * 0.06, source['xyz'][2] - k * 0.3])
                for receiver in data['receivers']:
                    lis_idx.append(int(re.findall(r'\d+', receiver['name'])[0]))
                    lis_locs.append(receiver['xyz'])
                src_lis_res = scene.computeIR(src_locs, lis_locs, ctx)

                for i_src in range(len(src_locs)):
                    for i_lis in range(len(lis_locs)):
                        save_path = os.path.join(save_folder, f'L{i + 1}_{j + 1}_{k + 1}_R{lis_idx[i_lis]:04}.wav')
                        audio_data = np.array(src_lis_res['samples'][i_src][i_lis])
                        stats = add_stat(stats, lis_locs[i_lis], src_locs[i_src], save_path)
                        with WaveWriter(save_path, channels=audio_data.shape[0],
                                        samplerate=int(src_lis_res['rate'])) as w:
                            w.write(audio_data)

    stats.to_csv(path_or_buf="stats.csv")

