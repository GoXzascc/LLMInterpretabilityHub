import argparse
import os
import yaml
import torch
import tqdm
import apprise
import transformer_lens
import mlflow
import safetensors
import json
import huggingface_hub
from loguru import logger
from dotenv import load_dotenv
from utils import seed_setup
from data import ActivationsStore
from sae import BatchTopKSAE, BaseAutoencoder, BatchAbsoluteKSAE, JumpReLUSAE
from functools import partial

load_dotenv()
# TODO: early stop

APPRISE_GMAIL = os.getenv("APPRISE_GMAIL")
APPRISE_PWD = os.getenv("APPRISE_PWD")
HF_TOKEN = os.getenv("HF_TOKEN")

huggingface_hub.login(HF_TOKEN)


def config() -> argparse.Namespace:
    """
    Config setup

    Returns:
        args: argparse.Namespace
    """
    parser = argparse.ArgumentParser()

    # basic information
    parser.add_argument(
        "--dataset",
        type=str,
        default="monology/pile-uncopyrighted",
        choices=["Skylion007/openwebtext", "monology/pile-uncopyrighted"],
    )
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--config_save_path", type=str, default="./configs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", type=bool, default=True)

    # SAE specific
    parser.add_argument(
        "--sae_name",
        type=str,
        default="batchabsolutek",
        choices=["batchabsolutek", "batchtopk", "jumprelu"],
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--aux_penalty", type=float, default=1 / 32)
    parser.add_argument("--k", type=int, default=51)
    parser.add_argument("--lr", type=int, default=3e-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--input_unit_norm", type=bool, default=True)
    parser.add_argument("--dictionary_factor", type=int, default=16)
    parser.add_argument("--bandwidth", type=float, default=0.001)
    parser.add_argument("--l1_coeff", type=float, default=0.004)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_batches_in_buffer", type=int, default=10)
    parser.add_argument("--num_tokens", type=int, default=int(1e9))
    parser.add_argument("--checkpoint_freq", type=int, default=5000)
    parser.add_argument("--training_steps", type=int, default=30000)
    parser.add_argument(
        "--n_batches_to_dead",
        type=int,
        default=5,
        help="Number of batches to consider a feature dead",
    )
    parser.add_argument(
        "--top_k_aux",
        type=int,
        default=512,
        help="Number of top k activations to use for auxiliary loss",
    )
    parser.add_argument(
        "--perf_log_freq",
        type=int,
        default=1000,
        help="Frequency of performance logging",
    )

    ## Optimizer specific
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.99, help="Beta2 for Adam optimizer"
    )
    parser.add_argument("--max_grad_norm", type=float, default=100000)

    # model specific
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-70m",
        choices=[
            "EleutherAI/pythia-70m",
            "google/gemma-2-2b",
            "Qwen/Qwen3-4B",
            "openai-community/gpt2",
        ],
    )

    args = parser.parse_args()

    return args


def post_init_cfg(args: argparse.Namespace) -> dict:
    cfg = vars(args)
    if cfg["model_name"] == "openai-community/gpt2":
        cfg["hook_point"] = f"blocks.{cfg['layer']}.hook_resid_post"
        cfg["model_name"] = "gpt2"
    elif cfg["model_name"] == "Qwen/Qwen3-4B":
        cfg["hook_point"] = f"blocks.{cfg['layer']}.hook_resid_post"
    elif cfg["model_name"] == "google/gemma-2-2b":
        cfg["hook_point"] = f"blocks.{cfg['layer']}.hook_resid_post"
    elif cfg["model_name"] == "EleutherAI/pythia-70m":
        cfg["hook_point"] = f"blocks.{cfg['layer']}.hook_resid_post"
    else:
        raise ValueError(f"Invalid model name: {cfg['model_name']}")

    # Generate hook point name
    cfg["name"] = (
        f"{cfg['model_name'].split('/')[-1]}_{cfg['dataset'].split('/')[-1]}_{cfg['hook_point']}_{cfg['dictionary_factor']}_{cfg['sae_name']}_{cfg['k']}_{cfg['lr']}"
    )
    logger.info(f"Post-initializing configuration: {cfg}")
    logger.info(f"Generated hook point: {cfg['hook_point']}")

    return cfg


def save_args(args: argparse.Namespace) -> None:
    config_file_path = args.config_save_path
    if not os.path.exists(config_file_path):
        os.makedirs(config_file_path, exist_ok=True)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    config_file_path = os.path.join(
        config_file_path, f"{model_name}_{dataset_name}_{args.sae_name}SAE.yaml"
    )
    log_file_path = os.path.join(
        args.log_path, f"{model_name}_{dataset_name}_{args.sae_name}SAE.log"
    )

    with open(config_file_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, indent=2)
    logger.add(log_file_path, rotation="100 MB", retention="10 days")
    logger.info(f"Training SAE with configuration: {args}")
    logger.info(f"Configuration saved to {config_file_path}")
    return model_name, dataset_name


def SAETrainer() -> None:
    # Step 1: Config and logger setup
    args = config()
    model_name, dataset_name = save_args(args)
    if args.verbose:
        logger.info("Verbose mode is enabled")
    else:
        logger.info("Verbose mode is disabled")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA is not available")
        raise ValueError("CUDA is not available")
    else:
        logger.info(f"Using device: {device}")

    seed_setup(args.seed)

    # Step 2: Model setup
    torch_dtype = getattr(torch, args.torch_dtype)
    cfg = post_init_cfg(args)

    model = (
        transformer_lens.HookedTransformer.from_pretrained(cfg["model_name"])
        .to(torch_dtype)
        .to(cfg["device"])
    )

    cfg["act_size"] = model.cfg.d_model
    cfg["dict_size"] = cfg["dictionary_factor"] * cfg["act_size"]
    cfg["dtype"] = torch_dtype
    activations_store = ActivationsStore(model, cfg)

    if args.sae_name == "batchabsolutek":
        sae = BatchAbsoluteKSAE(cfg)
    elif args.sae_name == "batchtopk":
        sae = BatchTopKSAE(cfg)
    elif args.sae_name == "jumprelu":
        sae = JumpReLUSAE(cfg)
    else:
        raise ValueError(f"Invalid SAE name: {args.sae_name}")

    # Step 3: Training
    train_sae(sae, activations_store, model, cfg)

    notify_gmail(
        message=f"SAE{args.sae_name}_{model_name}_Layer{args.layer}_{args.dataset} training completed",
        subject=f"SAE Training Completed - {model_name} Layer {args.layer} {args.sae_name} SAE {args.dataset}",
    )
    return None


def train_sae(
    sae: BaseAutoencoder,
    activations_store: ActivationsStore,
    model: transformer_lens.HookedTransformer,
    cfg: dict,
) -> None:
    # num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
    )
    training_steps = cfg["training_steps"]
    pbar = tqdm.tqdm(range(training_steps))
    mlflow.set_experiment(cfg["name"])
    mlflow.pytorch.autolog()
    mlflow.log_params(cfg)

    for idx in pbar:
        batch = activations_store.next_batch()
        sae_output = sae(batch)
        mlflow.log_metrics(
            {
                "loss": sae_output["loss"].item(),
                "l0_norm": sae_output["l0_norm"].item(),
                "l2_loss": sae_output["l2_loss"].item(),
                "l1_loss": sae_output["l1_loss"].item(),
                "l1_norm": sae_output["l1_norm"].item(),
                "num_dead_features": sae_output["num_dead_features"].item(),
                "positive_features": sae_output["positive_features"].item(),
                "negative_features": sae_output["negative_features"].item(),
            },
            step=idx,
        )
        if idx % cfg["perf_log_freq"] == 0:
            log_model_performance(idx, model, activations_store, sae)
        if idx % cfg["checkpoint_freq"] == 0:
            save_checkpoint(sae, cfg, idx)

        loss = sae_output["loss"]
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "L0": f"{sae_output['l0_norm']:.4f}",
                "L2": f"{sae_output['l2_loss']:.4f}",
                "L1": f"{sae_output['l1_loss']:.4f}",
                "L1_norm": f"{sae_output['l1_norm']:.4f}",
            }
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(sae, cfg, training_steps)


# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out


def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)


def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)


@torch.no_grad()
def log_model_performance(
    idx: int,
    model: transformer_lens.HookedTransformer,
    activations_store: ActivationsStore,
    sae: BaseAutoencoder,
    index: int = None,
    batch_tokens: torch.Tensor = None,
) -> None:
    """Log the model performance

    Args:
        idx: int: Index of the batch
        model: transformer_lens.HookedTransformer: Model
        activation_store: ActivationsStore: Activations store
        sae: BaseAutoencoder: SAE
    """
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[
            : sae.cfg["batch_size"] // sae.cfg["seq_len"]
        ]
    batch = activations_store.get_activations(batch_tokens).reshape(
        -1, sae.cfg["act_size"]
    )

    sae_output = sae(batch)["sae_out"].reshape(
        batch_tokens.shape[0], batch_tokens.shape[1], -1
    )

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss)
        / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss)
        / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    mlflow.log_metrics(log_dict, step=idx)
    return None


def save_checkpoint(sae: BaseAutoencoder, cfg: dict, idx: int) -> None:
    """Save the checkpoint

    Args:
        sae: BaseAutoencoder: SAE
        cfg: dict: Configuration
        idx: int: Index of the batch
    """
    save_dir = f"checkpoints/{cfg['name']}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, f"sae_{idx}.safetensors")
    safetensors.torch.save_file(sae.state_dict(), sae_path)
    logger.info(f"Model saved as {sae_path}")
    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, f"config_{idx}.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)
    logger.info(f"Config saved as {config_path}")
    return None


def notify_gmail(message: str, subject: str = "SAE Training Notification") -> None:
    """
    Notify via gmail

    Args:
        message (str): Message to send
        subject (str): Subject of the email

    Returns:
        None
    """
    notifier = apprise.Apprise()
    notifier.add(f"mailto://{APPRISE_GMAIL}:{APPRISE_PWD}@gmail.com")

    notifier.notify(body=message, title=subject)
    return None


if __name__ == "__main__":
    SAETrainer()
