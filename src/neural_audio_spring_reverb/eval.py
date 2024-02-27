# ./src/eval.py

import os
import torch
import torchaudio
import torchaudio.functional as F
import wandb
import auraloss
from datetime import datetime
from pathlib import Path

from .data.egfxset import load_egfxset
from .data.springset import load_springset
from .data.customset import load_customset
from .networks.model_utils import load_model_checkpoint
from tqdm import tqdm


def evaluate_model(args):
    print("Evaluating model...")
    print(f"Using backend: {torchaudio.get_audio_backend()}")

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    # torch.cuda.empty_cache()

    model, _, _, config, rf, params = load_model_checkpoint(args)

    # Initialize WandB logger
    wandb.init(
        project="neural-audio-spring-reverb",
        name=config["name"],
        job_type="eval",
        config=config,
    )

    # Define metrics
    mae = torch.nn.L1Loss()
    esr = auraloss.time.ESRLoss()
    dc = auraloss.time.DCLoss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=config["sample_rate"],
        perceptual_weighting=True,
    )

    criterions = [mae, esr, dc, mrstft]
    test_results = {"eval/mae": [], "eval/esr": [], "eval/dc": [], "eval/mrstft": []}

    rtf_list = []

    # Load data
    if config["dataset"] == "egfxset":
        _, _, test_loader = load_egfxset(
            args.data_dir,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
    elif config["dataset"] == "springset":
        _, _, test_loader = load_springset(
            args.data_dir,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
    elif config["dataset"] == "customset":
        _, _, test_loader = load_customset(
            args.data_dir,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
    else:
        raise ValueError("Dataset not found, options are: egfxset or springset")

    num_batches = len(test_loader)
    sr_tag = str(int(config["sample_rate"] / 1000)) + "k"
    label = f"{config['name']}-{config['criterion1']}-{sr_tag}"

    # Get the condition tensor
    if config["cond_dim"] > 0:
        c_values = [config.get(f"c{i}", 0.0) for i in range(config["cond_dim"])]
        c = (
            torch.tensor(c_values, device=args.device, requires_grad=False)
            .view(1, -1)
            .repeat(config["batch_size"], 1)
        )
    else:
        c = None

    model.eval()
    with torch.no_grad():
        for step, (dry, wet) in enumerate(
            tqdm(test_loader, total=num_batches, desc="Processing batches")
        ):
            start_time = datetime.now()
            global_step = step + 1

            input = dry.to(args.device)
            target = wet.to(args.device)

            pred = model(input, c)

            end_time = datetime.now()
            duration = end_time - start_time
            num_samples = input.size(-1) * config["batch_size"]
            lenght_in_seconds = num_samples / config["sample_rate"]
            rtf = duration.total_seconds() / lenght_in_seconds
            rtf_list.append(rtf)

            # Compute metrics means for current batch
            for metric, name in zip(criterions, test_results.keys()):
                batch_score = metric(pred, target).item()
                test_results[name].append(batch_score)

            # Save audios from last batch
            if step == num_batches - 1:
                # output = torchaudio.functional.highpass_biquad(output, sample_rate, 20)
                # target = torchaudio.functional.highpass_biquad(target, sample_rate, 20)

                input = input.view(-1).unsqueeze(0).cpu()
                target = target.view(-1).unsqueeze(0).cpu()
                pred = pred.view(-1).unsqueeze(0).cpu()

                input /= torch.max(torch.abs(input))
                target /= torch.max(torch.abs(target))
                pred /= torch.max(torch.abs(pred))

                os.makedirs(f"{args.audio_dir}/eval", exist_ok=True)

                # save_in = f"{args.audio_dir}/eval/input_{label}.wav"
                # torchaudio.save(save_in, input, config['sample_rate'])

                save_out = f"{args.audio_dir}/eval/pred-{label}.wav"
                torchaudio.save(save_out, pred, config["sample_rate"])

                save_target = f"{args.audio_dir}/eval/target-{label}.wav"
                torchaudio.save(save_target, target, config["sample_rate"])

    mean_test_results = {k: sum(v) / len(v) for k, v in test_results.items()}
    avg_rtf = sum(rtf_list) / len(rtf_list)
    mean_test_results["eval/rtf"] = avg_rtf

    wandb.log(mean_test_results)

    wandb.finish()
