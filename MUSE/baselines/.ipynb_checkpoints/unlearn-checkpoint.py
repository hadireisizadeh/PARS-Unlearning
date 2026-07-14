import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(BASELINE_PATH))

from baselines import unlearn_probe

import argparse
from os.path import basename, dirname, join as pathjoin


PROBE_ALGOS = {
    'probe_ga',
    'probe_ga_gdr',
    'probe_ga_klr',
    'probe_npo',
    'probe_npo_gdr',
    'probe_npo_klr',
}

def main():
    args = get_args()

    if args.algo in PROBE_ALGOS:
        base_loss = args.algo[len('probe_'):]
        unlearn_probe(
            model_dir=args.model_dir,
            data_file=args.data_file,
            out_dir=args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=base_loss,
            per_device_batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            retain_alpha=args.retain_alpha,
            probe_layers=args.probe_layers,
            probe_lr=args.probe_lr,
            probe_inner_steps=args.probe_inner_steps,
            probe_beta=args.probe_beta,   # --beta maps to probe_beta param
            probe_device=args.probe_device,
            keep_checkpoints=args.keep_checkpoints,
            probe_r_f_both=args.probe_r_f_both,
        )

    else:
        raise ValueError(
            f"Unknown algorithm '{args.algo}'. "
            f"Valid options: {sorted(PROBE_ALGOS)}"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Probe unlearning")

    all_algos = sorted(PROBE_ALGOS)
    parser.add_argument(
        '--algo', type=str, default='probe_npo_gdr',
        choices=all_algos,
        help=f"Unlearning algorithm. One of: {all_algos}. "
             "Default: probe_npo_gdr"
    )
    parser.add_argument(
        '--model_dir', type=str, default='muse-bench/MUSE-news_target',
        help="Path to the target model's HF directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default='meta-llama/Llama-2-7b-hf',
        help="Path to tokenizer's HF directory. Defaults to model_dir."
    )
    parser.add_argument(
        '--data_file', type=str, default='MUSE/data/news/raw/forget.txt',
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str, default='output',
        help="Path to output model directory."
    )
    parser.add_argument(
        '--max_len', type=int, default=2048,
        help="Max token length of inputs fed to the model."
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )
    parser.add_argument(
        '--per_device_batch_size', type=int, default=2,
    )
    parser.add_argument(
        '--retain_data_file', type=str, default='MUSE/data/news/raw/retain1.txt',
        help="Path to the retain set file. "
             "Required for *_gdr and *_klr variants."
    )
    parser.add_argument(
        '--retain_alpha', type=float, default=1.0,
        help="Weight applied to the retain loss term (gdr: retain CE loss; "
             "klr: retain KL term). Default: 1.0."
    )
    parser.add_argument(
        '--lr', type=float, default=2e-5,
        help="Model learning rate."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
    )

    parser.add_argument(
        '--probe_layers', type=int, nargs='+',
        default=[24,26,28,30],
        help="Layers at which to apply the probe probe loss. "
             "Use [8,10,12,14] for verbmem leakage (default). "
             "Use [28,29,30,31] for knowmem leakage."
    )
    parser.add_argument(
        '--probe_lr', type=float, default=1e-4,
        help="Learning rate for probe decoder inner maximization. "
             "Should be higher than --lr so probes track the model quickly."
    )
    parser.add_argument(
        '--probe_inner_steps', type=int, default=2,
        help="Number of inner gradient ascent steps per outer model step (k)."
    )
    parser.add_argument(
        '--probe_beta', type=float, default=0.5,
        help="Weight of the probe probe loss in the full objective."
    )
    parser.add_argument(
        '--probe_device', type=int, default=7,
        help="GPU device for probe (separate from model GPUs). "
             "Default: 7 (uses GPU 7 for probe while model uses 0-6)."
    )
    parser.add_argument(
        '--probe_r_f_both', action='store_true',
        help="Use both forget (+) and retain (-) in probe loss. "
             "When enabled, probe learns to decode forget data and NOT decode retain data."
    )

    parser.add_argument(
        '--keep_checkpoints', type=int, default=5,
        help="Number of checkpoints to keep on disk (save_total_limit)."
    )

    args = parser.parse_args()

    retain_required = (
        'gdr' in (args.algo or '') or
        'klr' in (args.algo or '')
    )
    if retain_required:
        assert args.retain_data_file is not None, \
            f"--retain_data_file is required for algo='{args.algo}'."

    return args


if __name__ == '__main__':
    main()