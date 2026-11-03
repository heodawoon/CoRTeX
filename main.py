import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['nsd', 'abide', 'cobre'])
    args, rest = ap.parse_known_args()

    sys.argv = [sys.argv[0]] + rest
    if args.task == 'nsd':
        from training.train_cortex_nsd import main as nsd_main
        nsd_main()
    elif args.task == 'abide':
        from training.train_cortex_abide import main as abide_main
        abide_main()
    elif args.task == 'cobre':
        from training.train_cortex_cobre import main as cobre_main
        cobre_main()


if __name__ == '__main__':
    main()
