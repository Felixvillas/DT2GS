from smac_maps import get_map_params

map_lists = [
    "3m",
    "8m",
    "25m",
    "5m_vs_6m",
    "8m_vs_9m",
    "10m_vs_11m",
    "27m_vs_30m",
    "2s3z",
    "3s5z",
    "3s5z_vs_3s6z",
    "3s_vs_3z",
    "3s_vs_4z",
    "3s_vs_5z",
    "1c3s5z",
    "2m_vs_1z",
    "corridor",
    "6h_vs_8z",
    "2s_vs_1sc",
    "so_many_baneling",
    "bane_vs_bane",
    "2c_vs_64zg"
]
batch_size = 3200

def get_obs_storage():
    print("\n=====obs storage analysis=====\n")
    for map_name in map_lists:
        map_info = get_map_params(map_name=map_name)
        n_agents = map_info["n_agents"]
        n_enemies = map_info["n_enemies"]
        obs_size = 16
        storage = batch_size * n_agents * (n_agents+n_enemies) * obs_size * 4 / 1000 / 1000

        print("Map: {:>16s} | Storage: {:>10f}MB".format(map_name, storage))
    
def get_state_storage():
    print("\n=====state storage analysis=====\n")
    for map_name in map_lists:
        map_info = get_map_params(map_name=map_name)
        n_agents = map_info["n_agents"]
        n_enemies = map_info["n_enemies"]
        state_size = 20
        storage = batch_size * n_agents * (n_agents+n_enemies) * state_size * 4 / 1000 / 1000

        print("Map: {:>16s} | Storage: {:>10f}MB".format(map_name, storage))

def get_embedding_storage():
    print("\n=====storage analysis for actor and critic input=====\n")
    for map_name in map_lists:
        map_info = get_map_params(map_name=map_name)
        n_agents = map_info["n_agents"]
        n_enemies = map_info["n_enemies"]
        embedding_size = 32

        storage = batch_size * n_agents * (n_agents+n_enemies) * embedding_size * 4 / 1000 / 1000

        print("Map: {:>16s} | Storage: {:>10f}MB | Entity num: {:>3d}".format(map_name, storage, n_agents+n_enemies))


if __name__ == "__main__":
    # get_obs_storage()
    # get_state_storage()
    get_embedding_storage()