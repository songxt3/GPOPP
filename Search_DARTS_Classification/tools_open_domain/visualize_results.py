import os
import sys
import argparse
sys.path.append(os.getcwd())
from nas_lib.utils.utils_darts import load_model_from_last_population
from nas_lib.visualize.visualize_gentype_darts import plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--model_path', type=str, default='model_path', help='The model path')
    parser.add_argument('--model_name', type=str, default='model_name', help='The model path')
    parser.add_argument('--gen_no', type=str, default=15, help='name of output files')
    parser.add_argument('--save_dir', type=str,
                        help='name of save directory')
    args = parser.parse_args()

    genotype, _ = load_model_from_last_population(args.save_dir, args.gen_no)

    plot(genotype.normal, f"normal_{args.model_name}")
    plot(genotype.reduce, f"reduction_{args.model_name}")