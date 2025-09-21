import torch
import safetensors
from loguru import logger


class BaseAutoencoder(torch.nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = torch.nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = torch.nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    @classmethod
    def from_pretrained(cls, checkpoint_path):
        """Load the model from a checkpoint using safetensors

        Args:
            checkpoint_path: Checkpoint path

        Returns:
            BaseAutoencoder: BaseAutoencoder model
        """
        checkpoint = safetensors.torch.load_file(checkpoint_path)
        baseautoencoder = cls(checkpoint["cfg"])
        baseautoencoder.load_state_dict(checkpoint["state_dict"])
        return baseautoencoder

    def preprocess_input(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess the input

        Args:
            x: torch.Tensor: Input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Preprocessed input
        """
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std) -> torch.Tensor:
        """Postprocess the output

        Args:
            x_reconstruct: torch.Tensor: Reconstructed input
            x_mean: torch.Tensor: Mean of the input
            x_std: torch.Tensor: Standard deviation of the input

        Returns:
            torch.Tensor: Postprocessed output
        """
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Make the decoder weights and grad unit norm

        Returns:
            None
        """
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts) -> None:
        """Update the inactive features

        Args:
            acts: torch.Tensor: Activations

        Returns:
            None
        """
        self.num_batches_not_active += (acts.sum(0) < 1e-6).float()
        self.num_batches_not_active[acts.sum(0) > 1e-6] = 0


class BatchAbsoluteKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x) -> dict:
        """Forward pass

        Args:
            x: torch.Tensor: Input tensor

        Returns:
            dict: Output dictionary
        """
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = x_cent @ self.W_enc
        flat = acts.flatten()

        # AbsoluteKSAE implementation
        abs_flat = flat.abs()
        topk = torch.topk(abs_flat, self.cfg["k"] * x.shape[0], dim=-1)
        mask_flat = torch.zeros_like(flat)
        mask_flat.scatter_(-1, topk.indices, 1.0)
        acts_topk = (flat * mask_flat).view_as(acts)

        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output
    
    def update_inactive_features(self, acts) -> None:
        """Update the inactive features

        Args:
            acts: torch.Tensor: Activations

        Returns:
            None
        """
        self.num_batches_not_active += (acts.abs().sum(0) < 1e-6).float()
        self.num_batches_not_active[acts.abs().sum(0) > 1e-6] = 0
        
    def get_loss_dict(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        acts: torch.Tensor,
        acts_topk: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> dict:
        """Get the loss dictionary

        Args:
            x: torch.Tensor: Input tensor
            x_reconstruct: torch.Tensor: Reconstructed input
            acts: torch.Tensor: Activations
            acts_topk: torch.Tensor: Activations topk
            x_mean: torch.Tensor: Mean of the input
            x_std: torch.Tensor: Standard deviation of the input

        Returns:
            dict: Output dictionary
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm

        # AbsoluteKSAE implementation
        l0_norm = (acts_topk.abs() > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        # loss = l2_loss + l1_loss + aux_loss
        loss = l2_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "positive_features": (acts_topk > 0).float().sum(-1).mean(),
            "negative_features": (acts_topk < 0).float().sum(-1).mean(),
        }
        return output

    def get_auxiliary_loss(
        self, x: torch.Tensor, x_reconstruct: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
        """Get the auxiliary loss

        Args:
            x: torch.Tensor: Input tensor
            x_reconstruct: torch.Tensor: Reconstructed input
            acts: torch.Tensor: Activations

        Returns:
            torch.Tensor: Auxiliary loss
        """
        # TODO: improve this
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            
            # act: [batch_size, dict_size]
            acts_abs = acts[:, dead_features].abs()
            acts_topk_aux = torch.topk(
                acts_abs,
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            mask_acts = torch.zeros_like(acts[:, dead_features])
            mask_acts.scatter_(-1, acts_topk_aux.indices, 1.0)
            acts_aux = (acts[:, dead_features] * mask_acts).view_as(acts[:, dead_features])
            
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x) -> dict:
        """Forward pass

        Args:
            x: torch.Tensor: Input tensor

        Returns:
            dict: Output dictionary
        """
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = torch.nn.functional.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        # loss = l2_loss + l1_loss + aux_loss
        loss = l2_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "positive_features": (acts_topk > 0).float().sum(-1).mean(),
            "negative_features": (acts_topk < 0).float().sum(-1).mean(),
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class RectangleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input
    
class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
    
    
class JumpReLU(torch.nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = torch.nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)
    
    

class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
   
class JumpReLUSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"], device=cfg["device"])
        self.dtype = cfg["dtype"]
        self.device = cfg["device"]

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)
        feature_magnitudes = feature_magnitudes.to(self.dtype)
        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
            "positive_features": (acts > 0).float().sum(-1).mean(),
            "negative_features": (acts < 0).float().sum(-1).mean(),
        }
        return output