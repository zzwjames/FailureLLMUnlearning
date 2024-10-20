# from .utils import load_model_and_tokenizer, load_model
# from .dataset import ForgetRetainDataset

# import torch
# import torch.nn.functional as F
# from torch.cuda import device_count
# import transformers
# from transformers import Trainer, AutoModelForCausalLM

# # model_dir is the address of learned target model
# # data_file is the address of the forget set
# # out_dir is the address of the unlearned model
# # retain_data_file is the address of the retain set

# def unlearn(
#     model_dir: str,
#     data_file: str,
#     out_dir: str,
#     retain_data_file: str | None = None,
#     loss_type: str = 'ga',
#     per_device_batch_size: int = 2,
#     epochs: int = 5,
#     learning_rate=1e-5,
#     max_len: int = 4096,
#     tokenizer_dir: str | None = None,
#     resume_from_checkpoint: bool = False
# ):
#     if 'gd' in loss_type:
#         assert retain_data_file is not None, "Retain data must be specified for grad_diff."
#     # load target learned model, here the model is finetuned on Llama2, so we load its tokenizer
#     model, tokenizer = load_model_and_tokenizer(
#         model_dir,
#         tokenizer_dir=tokenizer_dir
#     )

#     # load reference model (learned model) for negative preference optimization and KL divergence constraints
#     ref_model = (
#         load_model(model_dir)
#         if 'npo' in loss_type or 'kl' in loss_type
#         else None
#     )
#     dataset = ForgetRetainDataset(
#         data_file,
#         tokenizer=tokenizer,
#         retain_file_path=retain_data_file,
#         max_len=max_len
#     )

#     if device_count() == 0:
#         raise ValueError("Device not detected!")

#     training_args = transformers.TrainingArguments(
#         output_dir=out_dir,
#         per_device_train_batch_size=per_device_batch_size,
#         learning_rate=learning_rate,
#         save_strategy='epoch',  # Save every epoch
#         num_train_epochs=epochs,
#         optim='adamw_torch',
#         lr_scheduler_type='constant',
#         bf16=True,
#         report_to='none'        # Disable wandb
#     )

#     trainer = IterativeUnlearner(
#         model=model,
#         ref_model=ref_model,
#         tokenizer=tokenizer,
#         train_dataset=dataset,
#         args=training_args,
#         data_collator=dataset.get_collate_fn(),
#         loss_type=loss_type
#     )
#     model.config.use_cache = False  # silence the warnings.
#     trainer.train(resume_from_checkpoint=resume_from_checkpoint)
#     trainer.save_model(out_dir)



# class IterativeUnlearner(Trainer):
#     """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
#     """

#     def __init__(self, *args,
#                  loss_type: str = 'ga',
#                  ref_model: AutoModelForCausalLM | None = None,
#                  beta: float = 0.1,
#                  **kwargs):
#         self.loss_type = loss_type
#         self.ref_model = ref_model
#         self.beta = beta    # Only relevant when `'po' in self.loss_type`

#         if ref_model is not None:
#             assert 'po' in self.loss_type or 'kl' in self.loss_type
#             ref_model = ref_model.eval()

#         super().__init__(*args, **kwargs)

#     # the input is get_collate_fn() from ForgetRetainDataset
#     def compute_loss(self, model, x, return_outputs=False):
#         """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
#         """
        
#         ### 1. Run model ###
#         x_f, x_r = x
#         outputs_f = model(
#             x_f['input_ids'],
#             labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
#             attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
#         )
#         loss_f = outputs_f.loss

#         if 'gdr' in self.loss_type or 'klr' in self.loss_type:
#             outputs_r = model(
#                 x_r['input_ids'],
#                 labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
#                 attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
#             )
#             loss_r = outputs_r.loss

#         if 'klf' in self.loss_type or 'npo' in self.loss_type:
#             with torch.no_grad():
#                 outputs_f_ref = self.ref_model(
#                     x_f['input_ids'],
#                     labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
#                     attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
#                 )

#         if 'klr' in self.loss_type:
#             with torch.no_grad():
#                 outputs_r_ref = self.ref_model(
#                     x_r['input_ids'],
#                     labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
#                     attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
#                 )

#         ### 2. Compute Loss ###
#         loss = 0

#         if 'ga' in self.loss_type:
#             loss += -loss_f

#         elif 'npo' in self.loss_type:
#             neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
#             loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

#         else:
#             raise NotImplementedError("Cannot infer the given loss type.")

#         if 'gdr' in self.loss_type:
#             loss += loss_r

#         if 'klf' in self.loss_type:
#             raise NotImplementedError("KL forget not implemented yet!")

#         if 'klr' in self.loss_type:
#             kl_r = F.kl_div(
#                 outputs_r.logits,
#                 outputs_r_ref.logits,
#                 reduction = 'batchmean',
#                 log_target = True
#             )
#             loss += kl_r

#         return (loss, outputs_f) if return_outputs else loss


#     def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
#         input_ids, labels, attention_mask = x
#         # forward pass
#         with torch.no_grad():
#             outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
#             logits = outputs.logits
#             loss = outputs.loss
#         return (loss, logits, labels)


from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
import numpy as np

# model_dir is the address of learned target model
# data_file is the address of the forget set
# out_dir is the address of the unlearned model
# retain_data_file is the address of the retain set

def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    alpha: float = 1.0,
    threshold: int = 90
):
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."
    # load target learned model, here the model is finetuned on Llama2, so we load its tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    # load reference model (learned model) for negative preference optimization and KL divergence constraints
    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type
        else None
    )
    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',  # Save every epoch
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none'        # Disable wandb
    )
    print('alpha', alpha)
    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        alpha=alpha,
        threshold=threshold
    )
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)


#### normal iterative unlearner ####
class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 alpha: float = 1.0,
                 threshold: int = 90,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when `'po' in self.loss_type`
        self.alpha = alpha  # Weighting for retain data loss
        self.threshold = threshold
        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)

    # the input is get_collate_fn() from ForgetRetainDataset
    def compute_loss(self, model, x, return_outputs=False):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        
        ### 1. Run model ###
        x_f, x_r = x
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

        ### 2. Compute Loss ###
        loss = 0

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            loss += loss_r * self.alpha

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction = 'batchmean',
                log_target = True
            )
            loss += kl_r * self.alpha

        return (loss, outputs_f) if return_outputs else loss


    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


from transformers import Trainer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np









# saliency map with layer level mask
# class IterativeUnlearner(Trainer):
#     """Custom Trainer for Unlearning with Saliency Map"""

#     def __init__(self, *args,
#                  loss_type: str = 'ga',
#                  ref_model: AutoModelForCausalLM | None = None,
#                  beta: float = 0.1,
#                  alpha: float = 1.0,  # Weighting for retain data loss
#                  threshold: int = 90,
#                  **kwargs):
#         self.loss_type = loss_type
#         self.ref_model = ref_model
#         self.beta = beta    # Only relevant when 'npo' in self.loss_type
#         self.alpha = alpha  # Weighting for retain data loss
#         self.threshold = threshold
#         self.threshold = 99
#         if ref_model is not None:
#             assert 'npo' in self.loss_type or 'klr' in self.loss_type, \
#                 "ref_model should be provided when 'npo' or 'klr' is in loss_type"
#             self.ref_model.eval()

#         super().__init__(*args, **kwargs)

#     def compute_loss(self, model, x, return_outputs=False):
#         x_f, x_r = x

#         loss_components = self.loss_type.split('_')

#         loss = 0
#         outputs_f = outputs_r = None

#         # Reset saliency mask
#         self.m_S = None

#         ### Compute loss on forget data ###
#         if 'ga' in loss_components or 'npo' in loss_components:
#             # Compute loss on forget data
#             outputs_f = model(
#                 x_f['input_ids'],
#                 labels=x_f.get('labels', x_f['input_ids'].clone()),
#                 attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
#             )

#             if 'ga' in loss_components:
#                 # Gradient Ascent on forget data
#                 loss_f = -outputs_f.loss
#             elif 'npo' in loss_components:
#                 # NPO loss on forget data
#                 with torch.no_grad():
#                     outputs_f_ref = self.ref_model(
#                         x_f['input_ids'],
#                         labels=x_f.get('labels', x_f['input_ids'].clone()),
#                         attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
#                     )
#                 neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
#                 loss_f = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
#             else:
#                 raise ValueError("Unknown loss component for forget data.")

#             loss += loss_f

#             # Zero existing gradients
#             self.optimizer.zero_grad()

#             # Backward pass for loss_f to get gradients
#             loss_f.backward(retain_graph=True)

#             # Compute layer-wise gradient norms
#             layer_grad_norms = {}
#             for name, param in self.model.named_parameters():
#                 if param.grad is not None:
#                     # Compute the L2 norm of gradients for each layer
#                     grad_norm = param.grad.detach().data.norm(2).item()
#                     layer_name = '.'.join(name.split('.')[:-1])  # Get layer name without parameter name
#                     if layer_name in layer_grad_norms:
#                         layer_grad_norms[layer_name] += grad_norm
#                     else:
#                         layer_grad_norms[layer_name] = grad_norm

#             # Determine threshold gamma (e.g., 90th percentile of gradient norms)
#             grad_norms = list(layer_grad_norms.values())
#             gamma = np.percentile(grad_norms, self.threshold)
#             print('threshold', self.threshold)
#             print('gamma', gamma)

#             # Create saliency mask at layer level
#             self.m_S = {}
#             for layer_name, norm in layer_grad_norms.items():
#                 self.m_S[layer_name] = 1.0 if norm >= gamma else 0.0

#         else:
#             raise ValueError("No valid forget data loss component found in loss_type.")

#         ### Compute loss on retain data ###
#         if 'gdr' in loss_components or 'klr' in loss_components:
#             outputs_r = model(
#                 x_r['input_ids'],
#                 labels=x_r.get('labels', x_r['input_ids'].clone()),
#                 attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
#             )
#         print('alpha', self.alpha)
#         if 'gdr' in loss_components:
#             # Gradient Descent on retain data
#             loss_r = outputs_r.loss
#             loss += self.alpha * loss_r  # Use self.alpha to weight the retain data loss

#         if 'klr' in loss_components:
#             # KL Divergence Regularization on retain data
#             with torch.no_grad():
#                 outputs_r_ref = self.ref_model(
#                     x_r['input_ids'],
#                     labels=x_r.get('labels', x_r['input_ids'].clone()),
#                     attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
#                 )
#             kl_r = F.kl_div(
#                 F.log_softmax(outputs_r.logits, dim=-1),
#                 F.softmax(outputs_r_ref.logits, dim=-1),
#                 reduction='batchmean'
#             )
#             loss += self.alpha * kl_r

#         return (loss, outputs_f) if return_outputs else loss

#     def optimizer_step(self, optimizer, **kwargs):
#         # Apply layer-wise mask to gradients if m_S is defined
#         if hasattr(self, 'm_S') and self.m_S is not None:
#             for name, param in self.model.named_parameters():
#                 if param.grad is not None:
#                     layer_name = '.'.join(name.split('.')[:-1])
#                     mask_value = self.m_S.get(layer_name, 0.0)
#                     param.grad.mul_(mask_value)
#         optimizer.step()
#         optimizer.zero_grad()

# saliency map with neuron level mask
class IterativeUnlearner(Trainer):
    """Custom Trainer for Unlearning with Neuron-Level Saliency Map"""

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 alpha: float = 1.0,  # Weighting for retain data loss
                 threshold: int = 99,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when 'npo' in self.loss_type
        self.alpha = alpha  # Weighting for retain data loss
        self.threshold = 90
        if ref_model is not None:
            assert 'npo' in self.loss_type or 'klr' in self.loss_type, \
                "ref_model should be provided when 'npo' or 'klr' is in loss_type"
            self.ref_model.eval()

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, x, return_outputs=False):
        x_f, x_r = x

        loss_components = self.loss_type.split('_')

        loss = 0
        outputs_f = outputs_r = None

        # Reset saliency mask
        self.m_S = None

        ### Compute loss on forget data ###
        if 'ga' in loss_components or 'npo' in loss_components:
            # Compute loss on forget data
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f.get('labels', x_f['input_ids'].clone()),
                attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
            )

            if 'ga' in loss_components:
                # Gradient Ascent on forget data
                loss_f = -outputs_f.loss
            elif 'npo' in loss_components:
                # NPO loss on forget data
                with torch.no_grad():
                    outputs_f_ref = self.ref_model(
                        x_f['input_ids'],
                        labels=x_f.get('labels', x_f['input_ids'].clone()),
                        attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
                    )
                neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
                loss_f = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            else:
                raise ValueError("Unknown loss component for forget data.")

            loss += loss_f

            # Zero existing gradients
            self.optimizer.zero_grad()

            # Backward pass for loss_f to get gradients
            loss_f.backward(retain_graph=True)

            # Compute neuron-wise gradient norms
            neuron_grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach().data.float()  # Cast to float32
                    if grad.dim() > 1:
                        # Compute the gradient norm per neuron along the first dimension
                        grad_norms_per_neuron = grad.norm(2, dim=list(range(1, grad.dim()))).cpu().numpy()
                    else:
                        # For 1D parameters (e.g., biases)
                        grad_norms_per_neuron = grad.abs().cpu().numpy()

                    for idx, grad_norm in enumerate(grad_norms_per_neuron):
                        neuron_name = f"{name}.{idx}"
                        neuron_grad_norms[neuron_name] = grad_norm

            # Determine threshold gamma (e.g., 90th percentile of gradient norms)
            grad_norms = list(neuron_grad_norms.values())
            gamma = np.percentile(grad_norms, self.threshold)
            print('threshold', self.threshold)
            print('gamma', gamma)

            # Create saliency mask at neuron level
            self.m_S = {neuron_name: 1.0 if norm >= gamma else 0.0 for neuron_name, norm in neuron_grad_norms.items()}

        else:
            raise ValueError("No valid forget data loss component found in loss_type.")

        ### Compute loss on retain data ###
        if 'gdr' in loss_components or 'klr' in loss_components:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r.get('labels', x_r['input_ids'].clone()),
                attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
            )
        print('alpha', self.alpha)
        if 'gdr' in loss_components:
            # Gradient Descent on retain data
            loss_r = outputs_r.loss
            loss += self.alpha * loss_r  # Use self.alpha to weight the retain data loss

        if 'klr' in loss_components:
            # KL Divergence Regularization on retain data
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r.get('labels', x_r['input_ids'].clone()),
                    attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
                )
            kl_r = F.kl_div(
                F.log_softmax(outputs_r.logits, dim=-1),
                F.softmax(outputs_r_ref.logits, dim=-1),
                reduction='batchmean'
            )
            loss += self.alpha * kl_r

        return (loss, outputs_f) if return_outputs else loss

    def optimizer_step(self, optimizer, **kwargs):
        # Apply neuron-wise mask to gradients if m_S is defined
        if hasattr(self, 'm_S') and self.m_S is not None:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    if grad.dim() > 1:
                        # Build the mask tensor per neuron
                        neuron_mask_values = [self.m_S.get(f"{name}.{idx}", 0.0) for idx in range(grad.shape[0])]
                        mask_shape = [grad.shape[0]] + [1]*(grad.dim()-1)
                        mask = torch.tensor(neuron_mask_values, device=grad.device, dtype=grad.dtype).view(*mask_shape)
                        grad.mul_(mask)
                    else:
                        # For 1D parameters (e.g., biases)
                        neuron_mask_values = [self.m_S.get(f"{name}.{idx}", 0.0) for idx in range(grad.shape[0])]
                        mask = torch.tensor(neuron_mask_values, device=grad.device, dtype=grad.dtype)
                        grad.mul_(mask)
        optimizer.step()
        optimizer.zero_grad()

