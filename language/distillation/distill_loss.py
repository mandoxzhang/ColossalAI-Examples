import math

import torch
import torch.nn as nn


class KLDLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super(KLDLoss, self).__init__()
        self.temperature = temperature
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

    def forward(
        self,
        logtis_input: torch.Tensor,
        logtis_target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logtis_x (torch.Tensor): [batch_size, label_num] or [batch_size, sequence_size, label_num]
            logtis_y (torch.Tensor): [batch_size, label_num] or [batch_size, sequence_size, label_num]
        """
        if logtis_input.shape != logtis_target.shape:
            raise ValueError(
                "logits_x shape {} not equal to logits_y shape {}".format(
                    logtis_input.shape, logtis_target.shape
                )
            )
        if mask is None:
            mask = logtis_input.new_ones(logtis_input.size(0), logtis_input.size(1))

        logtis_input = logtis_input / self.temperature
        logtis_target = logtis_target / self.temperature
        loss = (
            self.kl_loss(
                torch.log_softmax(logtis_input, dim=-1),
                torch.softmax(logtis_target, dim=-1),
            )
            * self.temperature
            * self.temperature
        )
        if logtis_input.dim() == 2:
            loss = loss.sum() / logtis_input.size(0)
        elif logtis_input.dim() == 3:
            loss = loss.sum(-1)  # [b, s]
            mask = mask.to(dtype=logtis_input.dtype)  # [b, s]
            loss = loss * mask
            loss = loss.sum(dim=-1) / mask.sum(dim=-1)
            loss = loss.mean()
        return loss


class PatienceLoss(nn.Module):
    def __init__(self, is_normalized: bool = False) -> None:
        super(PatienceLoss, self).__init__()
        self.is_normalized = is_normalized
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(
        self,
        student_hidden_state: torch.Tensor,
        teacher_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logtis_x (torch.Tensor): [batch_size, hidden_size] or [batch_size, sequence_size, hidden_size]
            logtis_y (torch.Tensor): [batch_size, hidden_size] or [batch_size, sequence_size, hidden_size]
        """
        if student_hidden_state.shape != teacher_hidden_state.shape:
            raise ValueError(
                "student_hidden_state shape {} not equal to teacher_hidden_state shape {}".format(
                    student_hidden_state.shape, teacher_hidden_state.shape
                )
            )
        if student_hidden_state.dim() != 2 and student_hidden_state.dim() != 3:
            raise ValueError(
                "student_hidden_state dim should be 2 or 3, but is {}".format(
                    student_hidden_state.dim()
                )
            )
        if student_hidden_state.dim() == 3:
            student_hidden_state = student_hidden_state[:, 0]  # cls hidden state
            teacher_hidden_state = teacher_hidden_state[:, 0]

        if self.is_normalized:
            student_hidden_state_norm = torch.norm(
                student_hidden_state, p=2, dim=1, keepdim=True
            )
            teacher_hidden_state_norm = torch.norm(
                teacher_hidden_state, p=2, dim=1, keepdim=True
            )
            student_hidden_state = torch.div(
                student_hidden_state, student_hidden_state_norm
            )
            teacher_hidden_state = torch.div(
                teacher_hidden_state, teacher_hidden_state_norm
            )
        batch_size = student_hidden_state.size(0)
        return self.mse_loss(student_hidden_state, teacher_hidden_state) / batch_size


class SelfAttentionRelationLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super(SelfAttentionRelationLoss, self).__init__()
        self.temperature = temperature
        self.kd_loss = KLDLoss()

    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype):
        extended_attention_mask = mask.unsqueeze(1)  # 在head index上面扩一个纬度
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        student_sequence_state_x: torch.Tensor,
        student_sequence_state_y: torch.Tensor,
        teacher_sequence_state_x: torch.Tensor,
        teacher_sequence_state_y: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_sequence_state (torch.Tensor): [batch_size, head_num, sequence_size, hidden_size]
            teacher_sequence_state (torch.Tensor): [batch_size, head_num, sequence_size, hidden_size]
        """
        if student_sequence_state_x.size()[:3] != teacher_sequence_state_x.size()[:3]:
            raise ValueError(
                "The first two dimensional shapes are inconsistent, student_sequence_state shape {}， teacher_sequence_state shape {}".format(
                    student_sequence_state_x.shape, teacher_sequence_state_x.shape
                )
            )
        (
            batch_size,
            head_num,
            sequence_length,
            student_hidden_size,
        ) = student_sequence_state_x.size()
        student_hidden_size = student_sequence_state_x.size(-1)
        teacher_hidden_size = teacher_sequence_state_x.size(-1)

        # [batch, head_num, seq, hidden] -> [batch * head_num, seq, hidden]
        student_sequence_state_x = student_sequence_state_x.reshape(
            (-1, sequence_length, student_hidden_size)
        )
        student_sequence_state_y = student_sequence_state_y.reshape(
            (-1, sequence_length, student_hidden_size)
        )
        teacher_sequence_state_x = teacher_sequence_state_x.reshape(
            (-1, sequence_length, teacher_hidden_size)
        )
        teacher_sequence_state_y = teacher_sequence_state_y.reshape(
            (-1, sequence_length, teacher_hidden_size)
        )

        student_value_rel = torch.bmm(
            student_sequence_state_x, student_sequence_state_y.transpose(1, 2)
        ) / math.sqrt(student_hidden_size)
        teacher_value_rel = torch.bmm(
            teacher_sequence_state_x, teacher_sequence_state_y.transpose(1, 2)
        ) / math.sqrt(teacher_hidden_size)
        if mask is None:
            mask = student_sequence_state_x.new_ones(batch_size, sequence_length)
        mask = mask.unsqueeze(1).expand(batch_size, head_num, sequence_length)
        mask = mask.reshape(-1, sequence_length)
        attention_mask = self._expand_mask(mask, student_value_rel.dtype)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        student_value_rel = student_value_rel + attention_mask
        teacher_value_rel = teacher_value_rel + attention_mask

        loss = self.kd_loss(student_value_rel, teacher_value_rel, mask)
        loss = loss * head_num  # loss最后是除以 head_num * batch_size

        return loss
