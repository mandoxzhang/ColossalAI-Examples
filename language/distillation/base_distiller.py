from abc import ABC, abstractmethod

import torch
import re
from torch import nn
from typing import Dict, Any, List, Tuple


class MetaDistiller(ABC):
    def __init__(self):
        self.student_intermediate_state = {}
        self.teacher_intermediate_state = {}

    def _hook_student_module_output(self, name: str):
        def hook_module_output(module, input, output):
            self.student_intermediate_state[name].append(output)

        return hook_module_output

    def _hook_teacher_module_output(self, name: str):
        def hook_module_output(module, input, output):
            self.teacher_intermediate_state[name].append(output)

        return hook_module_output

    @abstractmethod
    def register_teacher_module_hook():
        """注册需要捕获teacher的中间输出的钩子"""
        pass

    @abstractmethod
    def register_student_module_hook():
        """注册需要捕获student的中间输出的钩子"""
        pass

    @abstractmethod
    def get_teacher_intermediate_states():
        """从cache好的dict解析出teacher需要intermediate_states"""
        pass

    @abstractmethod
    def get_student_intermediate_states():
        """从cache好的dict解析出student需要intermediate_states"""
        pass

    @abstractmethod
    def compute_loss(
        self,
        student_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        teacher_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
    ):
        """输入student和teacher模型的原始输出。结合get_intermediate_states，负责计算最后的蒸馏loss

        Args:
            student_output_tuple_list (Tuple[Loss: torch.Tensor, Logits_list: Tuple[torch.Tensor]]): student模型的原始输出
            teacher_output_tuple_list (Tuple[Loss: torch.Tensor, Logits_list: Tuple[torch.Tensor]]): teacher模型的原始输出

        """
        pass
