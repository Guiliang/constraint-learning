import os

from utils.data_utils import read_args, load_config


def generate_data_for_environment(args):
    config, debug_mode, log_file_path = load_config(args)

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:
        debug_msg = 'debug-'

    for data_path in os.listdir(config['source_path']):
        pass


if __name__ == "__main__":
    args = read_args()

    generate_data_for_environment(args)
