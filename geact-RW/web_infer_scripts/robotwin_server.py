"""GE-act openpi-protocol policy server for the RoboTwin 50-task campaign.

Thin subclass of the stock MVActorServer (web_infer_utils/server.py) that adds
per-request statistics-domain switching: the RoboTwin action post-training
(configs/ltx_model/robotwin/action_model_robotwin_{base,align_bidir}.yaml)
normalized actions/states with PER-DOMAIN mean/std from
robotwin_clean50_stats.json (keys '<domain>_eef' / '<domain>_state_eef',
domain = '<task>-demo_clean_collect_200-50', except
'put_bottles_dustbin-piper_clean_50-50').  One server process serves all 50
tasks, so the RoboTwin sim client sends 'domain_name' in every request and we
swap the de-normalization tensors when it changes.

Request message (msgpack-numpy dict, keys = MVActor.play kwargs + domain_name):
    obs             : uint8 [3, 256, 256, 3] RGB, cams [cam_high, cam_left_wrist,
                      cam_right_wrist] resized to the training sample_size 256x256
    state           : float [32] = concat(zeros(16), raw_endpose16)
                      endpose16 = [xyz, quat_wxyz, gripper01] x {left, right}
    state_zeropadding : [16, 0]   (history token = [zeros(act16); norm(state16)],
                      matching pack_action_state training layout)
    ndim_action     : 16          (de-norm/return only the true action channels)
    execution_step  : int         (how many of the 36 predicted actions to return)
    prompt          : str, prefix '<reset>' on the first call of each episode
    domain_name     : str stats key prefix for this task
Response: dict(actions=float32 [execution_step, 16] absolute eef actions in sim
convention -- directly consumable by TASK_ENV.take_action(a, action_type='ee')).
"""
import os
import sys
import socket
import logging
import argparse

import numpy as np
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from web_infer_utils.server import MVActorServer


class RoboTwinMVActorServer(MVActorServer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_domain = kwargs.get("domain_name", None)
        self._logged_state_check = False

    def _set_domain(self, domain_name):
        if domain_name == self._current_domain:
            return
        assert self.norm_type == "meanstd", "robotwin training used meanstd normalization"
        act_key = domain_name + "_" + self.action_space
        sta_key = domain_name + "_state_" + self.action_space
        assert act_key in self.StatisticInfo, f"missing stats key: {act_key}"
        assert sta_key in self.StatisticInfo, f"missing stats key: {sta_key}"
        self.act_mean = torch.tensor(self.StatisticInfo[act_key]["mean"]).unsqueeze(0).unsqueeze(0)
        self.act_std = torch.tensor(self.StatisticInfo[act_key]["std"]).unsqueeze(0).unsqueeze(0) + 1e-6
        self.sta_mean = np.array(self.StatisticInfo[sta_key]["mean"])
        self.sta_std = np.array(self.StatisticInfo[sta_key]["std"]) + 1e-6
        self._current_domain = domain_name
        self._logged_state_check = False
        print(f"[robotwin-server] stats domain -> {domain_name}", flush=True)

    def play(self, obs, prompt, domain_name=None, **kwargs):
        if domain_name is not None:
            self._set_domain(domain_name)
        # one-shot sanity check per domain: normalized state should be O(1)
        state = kwargs.get("state", None)
        if state is not None and not self._logged_state_check:
            pad = kwargs.get("state_zeropadding", [0, 0])
            raw = np.asarray(state, dtype=np.float64)
            core = raw[pad[0]:len(raw) - pad[1]] if pad[1] > 0 else raw[pad[0]:]
            normed = (core - self.sta_mean) / self.sta_std
            mx = float(np.abs(normed).max())
            print(f"[robotwin-server] domain={self._current_domain} "
                  f"|normed_state|_max={mx:.2f} (should be O(1); >8 suggests a "
                  f"state-convention mismatch)", flush=True)
            self._logged_state_check = True
        return super().play(obs=obs, prompt=prompt, **kwargs)


def get_args():
    parser = argparse.ArgumentParser(description="GE-act RoboTwin policy server")
    parser.add_argument('-c', '--config', type=str, required=True, help='eval server YAML')
    parser.add_argument('-w', '--weight', type=str, required=True,
                        help='path to diffusion_pytorch_model.safetensors')
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('-p', '--port', type=int, default=8001)
    parser.add_argument('--domain_name', type=str,
                        default="click_alarmclock-demo_clean_collect_200-50",
                        help='initial stats domain (client overrides per request)')
    parser.add_argument('--threshold', type=float, default=48,
                        help='executed steps between obs-memory rotations')
    parser.add_argument('--denoise_step', type=int, default=10)
    parser.add_argument('--action_dim', type=int, default=32,
                        help='packed action+state channels (16 act + 16 state)')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    policy_metadata = dict(test_meta="GE-act RoboTwin dual-arm action model (32ch packed)")

    actor = RoboTwinMVActorServer(
        args.host, args.port, policy_metadata,
        config_file=args.config,
        transformer_file=args.weight,
        load_weights=True,
        threshold=args.threshold,
        domain_name=args.domain_name,
        num_inference_steps=args.denoise_step,
        action_dim=args.action_dim,
        gripper_dim=1,
    )

    hostname = socket.gethostname()
    logging.info("Creating server (host: %s)", hostname)
    print(f"[robotwin-server] READY on {args.host}:{args.port} "
          f"(action_dim={args.action_dim}, denoise={args.denoise_step}, "
          f"threshold={args.threshold})", flush=True)
    actor.serve_forever()
