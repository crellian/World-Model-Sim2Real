import numpy as np
import os
root_dir = "/home2/random_bev_carla/rgb_bev/Town01_0"
for root, subdirs, files in os.walk(root_dir):
    if len(files) > 1:
        obs = np.load(os.path.join(root, "observation.npy"))
        rew = np.load(os.path.join(root, "reward.npy"))
        act = np.load(os.path.join(root, "action.npy"))
        ter = np.load(os.path.join(root, "terminal.npy"))

        indices = np.where(ter == 1)

        slices_a = []
        slices_r = []
        slices_v = []
        slice_epi = []
        slice_limit = []
        # Iterate through the indices and add slices to the lists
        start_idx = 0
        count = 0
        prev_idx = -1
        id_dict = {}
        for idx in indices[0]:
            slices_a.append(act[start_idx:idx+1])
            slices_r.append(rew[start_idx:idx+1])
            slice_epi += [count]*(idx - (prev_idx+1) + 1)
            slice_limit += [idx]*(idx - (prev_idx+1) + 1)
            id_dict[count] = start_idx
            #print(prev_idx, idx, len(slice_limit))
            assert(len(slice_epi) == len(slice_limit) == idx+1)
            assert(ter[len(slice_epi)-1] == 1)
            assert(ter[slice_limit[-1]] == 1)
            prev_idx = idx

            start_idx = idx+1
            count += 1
        print(id_dict)
        print(len(slice_epi))
        slice_epi += [count]*(rew.shape[0] - len(slice_epi))
        slice_limit += [(rew.shape[0]-1)]*(rew.shape[0] - len(slice_limit))
        assert(ter[len(slice_epi)-1] == 1)
        for abcd in range(rew.shape[0]):
            assert(ter[slice_limit[abcd]] == 1)
        assert(ter[slice_limit[-1]] == 1)

        np_epi = np.stack(slice_epi)
        np_limit = np.stack(slice_limit)

        np.save(os.path.join(root, "limit"), np_limit)
        np.save(os.path.join(root, "episode"), np_epi)
        np.save(os.path.join(root, "id_dict"), id_dict)
