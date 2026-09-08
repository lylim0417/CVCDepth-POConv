# Copyright (c) 2023 42dot. All rights reserved.
import time
from collections import defaultdict
from tqdm import tqdm
from kornia.geometry.transform import hflip
import torch
import torch.distributed as dist

from utils import Logger


class VFDepthTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        # if self.rank == 0:
        #     val_dataloader = model.val_dataloader()
        #     self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        for self.epoch in range(self.num_epochs):
            if self.ddp_enable:
                model.train_sampler.set_epoch(self.epoch) 
                
            self.train(model, train_dataloader, start_time)
            
            # save model after each epoch using rank 0 gpu 
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-'*110) 
                
            if self.ddp_enable:
                dist.barrier()

        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        Train one epoch.

        For single-GPU reproduction we optionally use gradient
        accumulation so that batch_size=1 can preserve the published
        effective optimizer batch without exercising the repository's
        batch_size>1 path.
        """
        model.set_train()

        accum_steps = int(
            getattr(self, 'gradient_accumulation_steps', 1)
        )

        if accum_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1"
            )

        num_batches = len(data_loader)

        if self.rank == 0:
            print(
                "Gradient accumulation:",
                accum_steps,
                "| dataloader batches:",
                num_batches,
            )

        model.optimizer.zero_grad(set_to_none=True)

        a = time.time()
        times = []

        # Scalar losses accumulated only for logging.
        scalar_loss_sums = {}

        group_start_time = time.time()

        for batch_idx, inputs in enumerate(data_loader):

            # Size of the current accumulation group.
            #
            # DDAD has 12319 samples, so the final group contains
            # 3 samples rather than 4. Divide that group by 3,
            # not by 4.
            group_start = (
                batch_idx // accum_steps
            ) * accum_steps

            group_size = min(
                accum_steps,
                num_batches - group_start
            )

            outputs, losses = model.process_batch(
                inputs,
                self.rank
            )

            total_loss = losses['total_loss']

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "Non-finite total loss at "
                    "batch {}".format(batch_idx)
                )

            # Average gradients over this effective optimizer batch.
            (
                total_loss / float(group_size)
            ).backward()

            # Collect scalar diagnostics for the effective batch.
            for key, value in losses.items():
                if (
                    torch.is_tensor(value)
                    and value.numel() == 1
                ):
                    scalar_loss_sums[key] = (
                        scalar_loss_sums.get(key, 0.0)
                        + float(value.detach())
                    )

            is_update_boundary = (
                ((batch_idx + 1) % accum_steps == 0)
                or
                ((batch_idx + 1) == num_batches)
            )

            if not is_update_boundary:
                continue

            # --------------------------------------------------
            # One optimizer update = one effective batch.
            # --------------------------------------------------
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

            after_op_time = time.time()

            if self.rank == 0:

                # Use effective-batch-average scalar losses in logs.
                log_losses = dict(losses)

                for key, value_sum in scalar_loss_sums.items():
                    log_losses[key] = torch.tensor(
                        value_sum / float(group_size),
                        device=total_loss.device,
                    )

                times.append(
                    after_op_time - group_start_time
                )

                print(
                    "batch",
                    batch_idx,
                    "| opt_step",
                    self.step,
                    "| group_size",
                    group_size,
                    "| elapsed",
                    sum(times),
                    "| group_time",
                    after_op_time - group_start_time,
                    "| avg sec/microbatch",
                    (time.time() - a) / (1 + batch_idx),
                )

                self.logger.update(
                    'train',
                    self.epoch,
                    self.world_size,
                    batch_idx,
                    self.step,
                    start_time,
                    group_start_time,
                    inputs,
                    outputs,
                    log_losses
                )

            if self.ddp_enable:
                dist.barrier()

            # Important: this now counts OPTIMIZER UPDATES,
            # not microbatches.
            self.step += 1

            scalar_loss_sums = {}
            group_start_time = time.time()

        # Preserve the author's epoch-level LR scheduling.
        model.lr_scheduler.step()

    @torch.no_grad()
    def validate(self, model):
        """
        This function validates models on validation dataset to monitor training process.
        """
        model.set_val()
        inputs = next(self.val_iter)
            
        outputs, losses = model.process_batch(inputs, self.rank)
        
        if 'depth' in inputs:
            depth_eval_metric, depth_eval_median = self.logger.compute_depth_losses(inputs, outputs, vis_scale=True)
            self.logger.print_perf(depth_eval_metric, 'metric')
            self.logger.print_perf(depth_eval_median, 'median')

        self.logger.log_tb('val', inputs, outputs, losses, self.step)            
        del inputs, outputs, losses
        
        model.set_train()
        
    @torch.no_grad()
    def evaluate(self, model,vis_results=False):
        """
        This function evaluates models on full validation dataset.
        """

        eval_dataloader = model.eval_dataloader()
        
        # load model
        model.load_weights()
        model.set_val()
        
        avg_depth_eval_metric = defaultdict(float)
        avg_depth_eval_median = defaultdict(float)

        avg_depth_eval_metric_cams = dict()
        for cam in range(self.num_cams):
            avg_depth_eval_metric_cams[cam] = defaultdict(float)
        process = tqdm(eval_dataloader)
        for batch_idx, inputs in enumerate(process):   
            # visualize synthesized depth maps
            if self.syn_visualize and batch_idx < self.syn_idx:
                continue
                
            outputs, _ = model.process_batch(inputs, self.rank)

            if hasattr(self,'post_process'):

                inputs[('color_aug', 0, 0)] = hflip(inputs[('color_aug', 0, 0)])
                outputs_flipped,_ = model.process_batch(inputs, self.rank)
                for cam in range(self.num_cams):
                    outputs['cam', cam][('depth', 0)] = self.batch_post_process_disparity_torch(outputs['cam', cam][('depth', 0)],hflip(outputs_flipped['cam', cam][('depth', 0)]))
                    # outputs['cam', cam][('depth', 0)] = self.post_process_inv_depth(outputs['cam', cam][('depth', 0)],outputs_flipped['cam', cam][('depth', 0)])

            depth_eval_metric, depth_eval_median,depth_eval_metric_cams = self.logger.compute_depth_losses(inputs, outputs)
            
            for key in self.depth_metric_names:
                avg_depth_eval_metric[key] += depth_eval_metric[key]
                avg_depth_eval_median[key] += depth_eval_median[key]

                for cam in range(self.num_cams):
                    avg_depth_eval_metric_cams[cam][key]+=depth_eval_metric_cams[cam][key]


            if vis_results:
                self.logger.log_result(inputs, outputs, batch_idx, self.syn_visualize)
            
            if self.syn_visualize and batch_idx >= self.syn_idx:
                process.close()
                break


        for key in self.depth_metric_names:
            avg_depth_eval_metric[key] /= len(eval_dataloader)
            avg_depth_eval_median[key] /= len(eval_dataloader)
            for cam in range(self.num_cams):
                avg_depth_eval_metric_cams[cam][key]/= len(eval_dataloader)

        print('Evaluation result...\n')
        self.logger.print_perf(avg_depth_eval_metric, 'metric')
        self.logger.print_perf(avg_depth_eval_median, 'median')
        return avg_depth_eval_metric,avg_depth_eval_median,avg_depth_eval_metric_cams

    @torch.no_grad()
    def visualize(self, model, vis_results=True):
        """
        This function evaluates models on full validation dataset.
        """
        eval_dataloader = model.eval_dataloader()

        # load model
        model.load_weights()
        model.set_val()

        avg_depth_eval_metric = defaultdict(float)
        avg_depth_eval_median = defaultdict(float)

        avg_depth_eval_metric_cams = dict()
        for cam in range(self.num_cams):
            avg_depth_eval_metric_cams[cam] = defaultdict(float)
        process = tqdm(eval_dataloader)
        for batch_idx, inputs in enumerate(process):

            outputs, _ = model.process_batch(inputs, self.rank)
            depth_eval_metric, depth_eval_median, depth_eval_metric_cams = self.logger.compute_depth_losses(inputs,outputs)


            self.logger.log_result(inputs, outputs, batch_idx,depth_eval_metric_cams,False)

    def batch_post_process_disparity_torch(self,l_disp, r_disp):
        """Apply the disparity post-processing method as introduced in Monodepthv1
        """
        _, _, h, w = l_disp.shape
        m_disp = 0.5 * (l_disp + r_disp)
        _, l = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))
        l_mask = (1.0 - torch.clip(20 * (l - 0.05), 0, 1))[None, None, ...].cuda()
        # r_mask = l_mask[:, :, ::-1]
        r_mask = l_mask.flip(-1)
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def post_process_inv_depth(self,inv_depth, inv_depth_flipped, method='mean'):
        """
        Post-process an inverse and flipped inverse depth map
        Parameters
        ----------
        inv_depth : torch.Tensor [B,1,H,W]
            Inverse depth map
        inv_depth_flipped : torch.Tensor [B,1,H,W]
            Inverse depth map produced from a flipped image
        method : str
            Method that will be used to fuse the inverse depth maps
        Returns
        -------
        inv_depth_pp : torch.Tensor [B,1,H,W]
            Post-processed inverse depth map
        """
        B, C, H, W = inv_depth.shape
        inv_depth_hat = self.flip_lr(inv_depth_flipped)
        inv_depth_fused = self.fuse_inv_depth(inv_depth, inv_depth_hat, method=method)
        xs = torch.linspace(0., 1., W, device=inv_depth.device,
                            dtype=inv_depth.dtype).repeat(B, C, H, 1)
        mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
        mask_hat = self.flip_lr(mask)
        return mask_hat * inv_depth + mask * inv_depth_hat + \
            (1.0 - mask - mask_hat) * inv_depth_fused

    def fuse_inv_depth(self,inv_depth, inv_depth_hat, method='mean'):
        """
        Fuse inverse depth and flipped inverse depth maps
        Parameters
        ----------
        inv_depth : torch.Tensor [B,1,H,W]
            Inverse depth map
        inv_depth_hat : torch.Tensor [B,1,H,W]
            Flipped inverse depth map produced from a flipped image
        method : str
            Method that will be used to fuse the inverse depth maps
        Returns
        -------
        fused_inv_depth : torch.Tensor [B,1,H,W]
            Fused inverse depth map
        """
        if method == 'mean':
            return 0.5 * (inv_depth + inv_depth_hat)
        elif method == 'max':
            return torch.max(inv_depth, inv_depth_hat)
        elif method == 'min':
            return torch.min(inv_depth, inv_depth_hat)
        else:
            raise ValueError('Unknown post-process method {}'.format(method))

    def flip_lr(self,image):
        """
        Flip image horizontally
        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Image to be flipped
        Returns
        -------
        image_flipped : torch.Tensor [B,3,H,W]
            Flipped image
        """
        assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
        return torch.flip(image, [3])