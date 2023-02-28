import torch
from torch import nn
from typing import Dict, Any, List, Tuple, Union

from distill_modeling import ProjectModel
from distill_loss import KLDLoss, PatienceLoss, SelfAttentionRelationLoss
from base_distiller import MetaDistiller
from distill_hook import (
    register_bert_attention_and_value_state_hook,
    register_bert_hidden_state_hook,
)

# from utils import load_weight_from_state_dict
# from utils import LOGGER, load_weight_from_state_dict


class BasicDisiller(MetaDistiller):
    """
    最简单的蒸馏kd-loss + hard label loss
    """

    def __init__(
        self,
        distill_config: Any,
        teacher_model_config: Any,
        student_model_config: Any,
        teacher_model: nn.Module,
        student_model: nn.Module,
    ):
        is_init_from_teacher = (
            distill_config.is_init_from_teacher
            and student_model_config.hidden_size == teacher_model_config.hidden_size
        )
        if is_init_from_teacher:
            self.student_init_from_teacher(teacher_model.encoder, student_model.encoder)
        self.hard_target_weight = getattr(distill_config, "hard_target_weight", 1.0)
        self.soft_target_weight = getattr(distill_config, "soft_target_weight", 1.0)
        self.soft_target_inner_weight = getattr(
            distill_config, "soft_target_inner_weight", []
        )  # 假如模型有多个logits输出，那么对应着多个kd_loss, 这里可以指定权重
        self.kd_loss_fct = KLDLoss(temperature=distill_config.temperature)

    def clear_cache(self):
        pass

    def student_init_from_teacher(
        self, teacher_model: nn.Module, student_model: nn.Module
    ):
        """
        定义了student如何从teacher继承参数，目前很简单，直接继承前x层
        """
        pass
        # teacher_state_dict = teacher_model.state_dict().copy()
        # # only load encoder param
        # load_weight_from_state_dict(student_model, teacher_state_dict)
        # return

    def register_teacher_module_hook(self, model):
        """注册需要捕获teacher的中间输出的钩子"""
        pass

    def register_student_module_hook(self, model):
        """注册需要捕获student的中间输出的钩子"""
        pass

    def get_teacher_intermediate_states(self):
        """Basic distiller only distill the output logtis"""
        pass

    def get_student_intermediate_states(self):
        """Basic distiller only distill the output logtis"""
        pass

    def _compute_loss_implement(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        input_mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = None
        loss_dict = {}
        # compute kd loss
        for i, (student_logits, teacher_logits) in enumerate(
            zip(
                student_output["output_logits_list"],
                teacher_output["output_logits_list"],
            )
        ):
            kd_loss = self.kd_loss_fct(student_logits, teacher_logits, mask=input_mask)
            weight = 1.0
            if i < len(self.soft_target_inner_weight):
                weight = self.soft_target_inner_weight[i]
            if total_loss is None:
                total_loss = weight * kd_loss
            else:
                total_loss = total_loss + weight * kd_loss
        total_loss = self.soft_target_weight * total_loss
        loss_dict["kd_loss"] = total_loss

        # compute hard label loss
        hard_label_loss = student_output.get("loss", None)
        if hard_label_loss is not None:
            total_loss = total_loss + self.hard_target_weight * hard_label_loss
            loss_dict["ori_loss"] = hard_label_loss
        else:
            print("Error | hard_label_loss is none!")

        return total_loss, loss_dict

    def compute_loss(
        self,
        student_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        teacher_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        input_mask=None,
    ) -> torch.Tensor:
        # contruct teacher all output
        teacher_output_dict = {}
        teacher_output_dict["output_logits_list"] = [
            teacher_output_tuple_list.logits.detach()
        ]
        # contruct student all output
        student_output_dict = {}
        student_output_dict["loss"] = student_output_tuple_list[0]
        student_output_dict["output_logits_list"] = [student_output_tuple_list.logits]
        return self._compute_loss_implement(
            student_output_dict, teacher_output_dict, input_mask
        )


class PkdDistiller(MetaDistiller):
    """
    PKD-Bert实现： https://arxiv.org/abs/1908.09355
    """

    def __init__(
        self,
        distill_config: Any,
        teacher_model_config: Any,
        student_model_config: Any,
        teacher_model: nn.Module,
        student_model: nn.Module,
    ):
        super(PkdDistiller, self).__init__()
        self.clear_cache()
        self.mode = distill_config.mode_type
        is_init_from_teacher = (
            distill_config.is_init_from_teacher
            and (student_model_config.hidden_size == teacher_model_config.hidden_size)
            and (student_model_config.model_type == teacher_model_config.model_type)
        )
        if is_init_from_teacher:
            self.student_init_from_teacher(
                teacher_model.encoder, student_model.encoder
            )  # 目前仅考虑transformer encoder类的模型
        self.use_project = (
            student_model_config.hidden_size != teacher_model_config.hidden_size
        )
        if self.use_project:
            print("use projcet model !")
            print(
                student_model_config.hidden_size, " ", teacher_model_config.hidden_size
            )
            self.project_model = ProjectModel(
                student_model_config.hidden_size, teacher_model_config.hidden_size
            )

        self.kd_loss_fct = KLDLoss(temperature=distill_config.temperature)
        self.pkd_loss_fct = PatienceLoss(is_normalized=distill_config.is_pkd_normalized)
        self.hard_target_weight = getattr(distill_config, "hard_target_weight", 1.0)
        self.soft_target_weight = getattr(distill_config, "soft_target_weight", 1.0)
        self.soft_target_inner_weight = getattr(
            distill_config, "soft_target_inner_weight", []
        )
        self.pkd_weight = getattr(distill_config, "pkd_weight", 1.0)  # patient loss的权重

    def clear_cache(self) -> None:
        """
        重新初始化cache, 每次forward前执行，防止二次访问
        """
        self.teacher_intermediate_state = {
            "hidden_state_list": [],
        }  # 目前先只考虑单teacher
        self.student_intermediate_state = {
            "hidden_state_list": [],
        }  # cache the teacher and student intermediate_state
        return

    def register_student_module_hook(self, model: nn.Module) -> None:
        # TODO(@yutian) 修改为根据model type和特征名获取对应的register函数
        register_bert_hidden_state_hook(
            model, self._hook_student_module_output, "hidden_state_list"
        )
        return

    def register_teacher_module_hook(self, model: nn.Module) -> None:
        register_bert_hidden_state_hook(
            model, self._hook_teacher_module_output, "hidden_state_list"
        )
        return

    def student_init_from_teacher(
        self, teacher_model: nn.Module, student_model: nn.Module
    ) -> None:
        """
        定义了student如何从teacher继承参数，目前很简单，直接继承前x层
        """
        # teacher_state_dict = teacher_model.state_dict().copy()
        # # only load encoder param
        # load_weight_from_state_dict(student_model, teacher_state_dict)
        # return
        pass

    def get_teacher_intermediate_states(
        self,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        获取teacher的中间变量，pkd-bert是要拿到每一层的hidden_state, hidden_state对应输出层有多个输出， hidden_state位于第一个输出
        """
        res = {}
        res["hidden_state_list"] = [
            x[0].detach() for x in self.teacher_intermediate_state["hidden_state_list"]
        ]
        return res

    def get_student_intermediate_states(self):
        """
        获取student的中间变量，pkd-bert是要拿到每一层的hidden_state, hidden_state对应输出层有多个输出， hidden_state位于第一个输出
        """
        res = {}
        res["hidden_state_list"] = [
            x[0] for x in self.student_intermediate_state["hidden_state_list"]
        ]
        return res

    def _pkd_skip(
        self,
        student_hidden_state_list: List[torch.Tensor],
        teacher_hidden_state_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        pkd-bert paper中的pkd-last模式，比如student有6层，teacher有12层，
        则student的每1层对应teacher的每2层的后一层，去计算layer-to-layer loss
        e.g. student第0层 对应 teacher第1层，第3层对应 teacher第7层
        """
        teacher_layer_num = len(teacher_hidden_state_list)
        student_layer_num = len(student_hidden_state_list)
        if teacher_layer_num % student_layer_num != 0:
            print(
                "Error | teacher_layer_num {} \% student_layer_num {} != 0".format(
                    teacher_layer_num, student_layer_num
                )
            )
            return None
        total_pkd_loss = None
        multiple = int(teacher_layer_num / student_layer_num)
        for student_layer_index in range(student_layer_num):
            teacher_layer_index = student_layer_index * multiple + multiple - 1
            # print(multiple, student_layer_index, teacher_layer_index)
            cur_student_hidden_state = student_hidden_state_list[student_layer_index]
            if self.use_project:
                # print(cur_student_hidden_state.size())
                cur_student_hidden_state = self.project_model(cur_student_hidden_state)
            cur_teacher_hidden_state = teacher_hidden_state_list[teacher_layer_index]
            cur_pkd_loss = self.pkd_loss_fct(
                cur_student_hidden_state, cur_teacher_hidden_state
            )
            if student_layer_index == 0:
                total_pkd_loss = cur_pkd_loss
            else:
                total_pkd_loss += cur_pkd_loss
        # raise ValueError
        return total_pkd_loss

    def _pkd_last(
        self,
        student_hidden_state_list: List[torch.Tensor],
        teacher_hidden_state_list: List[torch.Tensor],
    ):
        """
        pkd-bert paper中的pkd-last模式，比如student有6层，teacher有12层，则student的6层对应最后6层去计算layer-to-layer loss
        """
        teacher_layer_num = len(teacher_hidden_state_list)
        student_layer_num = len(student_hidden_state_list)
        if teacher_layer_num % student_layer_num != 0:
            print(
                "Error | teacher_layer_num {} % tudent_layer_num {} != 0".format(
                    teacher_layer_num, student_layer_num
                )
            )
            return None
        total_pkd_loss = None
        for student_layer_index in range(student_layer_num):
            teacher_layer_index = (
                teacher_layer_num - student_layer_num + student_layer_index
            )
            cur_student_hidden_state = student_hidden_state_list[student_layer_index]
            if self.use_project:
                cur_student_hidden_state = self.project_model(cur_student_hidden_state)
            cur_teacher_hidden_state = teacher_hidden_state_list[teacher_layer_index]
            cur_pkd_loss = self.pkd_loss_fct(
                cur_student_hidden_state, cur_teacher_hidden_state
            )
            if student_layer_index == 0:
                total_pkd_loss = cur_pkd_loss
            else:
                total_pkd_loss = total_pkd_loss + cur_pkd_loss
        return total_pkd_loss

    def _compute_loss_implement(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        input_mask: torch.Tensor,
    ):
        total_loss = None
        loss_dict = {}
        # compute kd loss
        for i, (student_logits, teacher_logits) in enumerate(
            zip(
                student_output["output_logits_list"],
                teacher_output["output_logits_list"],
            )
        ):
            kd_loss = self.kd_loss_fct(student_logits, teacher_logits, mask=input_mask)
            weight = 1.0
            if i < len(self.soft_target_inner_weight):
                weight = self.soft_target_inner_weight[i]
            if total_loss is None:
                total_loss = weight * kd_loss
            else:
                total_loss = total_loss + weight * kd_loss
        total_loss = self.soft_target_weight * total_loss
        loss_dict["kd_loss"] = total_loss.item()

        # compute hard label loss
        hard_label_loss = student_output.get("loss", None)
        if hard_label_loss is not None:
            total_loss = total_loss + self.hard_target_weight * hard_label_loss
            loss_dict["ori_loss"] = hard_label_loss.item()
        else:
            print("Error | hard_label_loss is none!")

        # compute
        if self.mode == "pkd-skip":
            pkd_loss = self._pkd_skip(
                student_output["hidden_state_list"], teacher_output["hidden_state_list"]
            )
        elif self.mode == "pkd-last":
            pkd_loss = self._pkd_last(
                student_output["hidden_state_list"], teacher_output["hidden_state_list"]
            )
        if pkd_loss is not None:
            total_loss = total_loss + self.pkd_weight * pkd_loss
            loss_dict["pkd_loss"] = pkd_loss.item()
        else:
            print("Error | pkd_loss is none!")

        return (total_loss, loss_dict)

    def compute_loss(
        self,
        student_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        teacher_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        input_mask=None,
    ):
        # contruct teacher all output
        teacher_output_dict = {}
        teacher_output_dict["output_logits_list"] = list(teacher_output_tuple_list[1])
        teacher_intermediate_states = self.get_teacher_intermediate_states()
        teacher_output_dict.update(teacher_intermediate_states)
        # contruct student all output
        student_output_dict = {}
        student_output_dict["loss"] = student_output_tuple_list[0]
        student_output_dict["output_logits_list"] = list(student_output_tuple_list[1])
        student_intermediate_states = self.get_student_intermediate_states()
        student_output_dict.update(student_intermediate_states)
        return self._compute_loss_implement(
            student_output_dict, teacher_output_dict, input_mask
        )


class MinilmV1Distiller(MetaDistiller):
    def __init__(
        self,
        distill_config: Any,
        teacher_model_config: Any,
        student_model_config: Any,
        teacher_model: nn.Module,
        student_model: nn.Module,
    ):
        super(MinilmV1Distiller, self).__init__()
        self.clear_cache()
        is_init_from_teacher = (
            distill_config.is_init_from_teacher
            and (student_model_config.hidden_size == teacher_model_config.hidden_size)
            and (student_model_config.model_type == teacher_model_config.model_type)
        )
        if is_init_from_teacher:
            self.student_init_from_teacher(
                teacher_model.encoder, student_model.encoder
            )  # 目前仅考虑transformer encoder类的模型
        # self.use_project = (
        #     student_model_config.hidden_size != teacher_model_config.hidden_size
        # )
        # if self.use_project:
        #     print("use projcet model !")
        #     print(
        #         student_model_config.hidden_size, " ", teacher_model_config.hidden_size
        #     )
        #     self.project_model = ProjectModel(
        #         student_model_config.hidden_size, teacher_model_config.hidden_size
        #     )

        self.kd_loss_fct = KLDLoss(temperature=distill_config.temperature)
        self.attention_score_loss_fct = KLDLoss()
        self.rel_loss_fct = SelfAttentionRelationLoss()
        self.hard_target_weight = getattr(distill_config, "hard_target_weight", 1.0)
        self.soft_target_weight = getattr(distill_config, "soft_target_weight", 1.0)
        self.soft_target_inner_weight = getattr(
            distill_config, "soft_target_inner_weight", []
        )
        # self.pkd_weight = getattr(distill_config, "pkd_weight", 1.0)  # patient loss的权重

    def clear_cache(self) -> None:
        """
        重新初始化cache, 每次forward前执行，防止二次访问
        """
        self.teacher_intermediate_state = {
            "attention_layer_output_list": [],
            "value_state_list": [],
        }  # 目前先只考虑单teacher
        self.student_intermediate_state = {
            "attention_layer_output_list": [],
            "value_state_list": [],
        }  # cache the teacher and student intermediate_state
        return

    def register_student_module_hook(self, model: nn.Module) -> None:
        # TODO(@yutian) 修改为根据model type和特征名获取对应的register函数
        # register_bert_hidden_state_hook(
        #     model, self._hook_student_module_output, "value_state_list"
        # )
        # 这里优点疑问，value relation应该是以最初的value output算出来的，这里可以后面实验下
        register_bert_attention_and_value_state_hook(
            model, self._hook_student_module_output, "attention_layer_output_list"
        )
        return

    def register_teacher_module_hook(self, model: nn.Module) -> None:
        # register_bert_hidden_state_hook(
        #     model, self._hook_teacher_module_output, "value_state_list"
        # )
        register_bert_attention_and_value_state_hook(
            model, self._hook_teacher_module_output, "attention_layer_output_list"
        )
        return

    def student_init_from_teacher(
        self, teacher_model: nn.Module, student_model: nn.Module
    ) -> None:
        """
        定义了student如何从teacher继承参数，目前很简单，直接继承前x层
        """
        # teacher_state_dict = teacher_model.state_dict().copy()
        # # only load encoder param
        # load_weight_from_state_dict(student_model, teacher_state_dict)
        # return
        pass

    def get_teacher_intermediate_states(
        self,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        获取teacher的中间变量，pkd-bert是要拿到每一层的hidden_state, hidden_state对应输出层有多个输出， hidden_state位于第一个输出
        """
        res = {}
        # res["hidden_state_list"] = [
        #     x[0].detach() for x in self.teacher_intermediate_state["hidden_state_list"]
        # ]
        res["attention_score_list"] = [
            x[-2].detach()
            for x in self.teacher_intermediate_state["attention_layer_output_list"]
        ]
        res["value_state_list"] = [
            x[-1].detach()
            for x in self.teacher_intermediate_state["attention_layer_output_list"]
        ]
        # print(
        #     "inter state size",
        #     len(self.teacher_intermediate_state["attention_layer_output_list"]),
        # )
        # print("attention_score_list size", len(res["attention_score_list"]))
        # print("value_state_list size", len(res["value_state_list"]))

        return res

    def get_student_intermediate_states(self):
        """
        获取student的中间变量，pkd-bert是要拿到每一层的hidden_state, hidden_state对应输出层有多个输出， hidden_state位于第一个输出
        """
        res = {}
        # res["hidden_state_list"] = [
        #     x[0] for x in self.student_intermediate_state["hidden_state_list"]
        # ]
        res["attention_score_list"] = [
            x[-2]
            for x in self.student_intermediate_state["attention_layer_output_list"]
        ]
        res["value_state_list"] = [
            x[-1]
            for x in self.student_intermediate_state["attention_layer_output_list"]
        ]
        # print("student layer 11 val ", res["value_state_list"][-1].size())
        # print("student layer 11 val ", res["value_state_list"][-1][:2, :3, :3, :3])
        # print(
        #     "student layer 11 attn ",
        #     res["attention_score_list"][-1].size(),
        # )
        # print(
        #     "student layer 11 attn",
        #     res["attention_score_list"][-1][:2, :3, :3, :3],
        # )
        # print(
        #     "student state size",
        #     len(self.student_intermediate_state["attention_layer_output_list"]),
        # )
        # print("student attention_score_list size", len(res["attention_score_list"]))
        # print("student value_state_list size", len(res["value_state_list"]))
        return res

    def _compute_attention_loss(
        self,
        student_attentin_score_list: List[torch.Tensor],
        teacher_attentin_score_list: List[torch.Tensor],
        mask=None,
    ) -> torch.Tensor:
        """
        计算q-k, 即attention scores对应的match loss, 当前只实现蒸馏最后一层
        Args:
            student_attentin_score_list (List[torch.Tensor]): [[batch_size, head_num, sequence_size, sequence_size], ...]
            teacher_attentin_score_list (List[torch.Tensor]):[[batch_size, head_num, sequence_size, sequence_size], ...]
            mask: [batch_size, sequence]
        """

        student_last_attention_score = student_attentin_score_list[-1]
        teacher_last_attention_score = teacher_attentin_score_list[-1]
        batch_size, head_num, seq_len, seq_len = student_last_attention_score.size()
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, head_num, seq_len)
            mask = mask.reshape(-1, seq_len)
        student_last_attention_score = student_last_attention_score.view(
            -1, seq_len, seq_len
        )
        teacher_last_attention_score = teacher_last_attention_score.view(
            -1, seq_len, seq_len
        )
        attention_match_kd_loss = self.attention_score_loss_fct(
            student_last_attention_score, teacher_last_attention_score, mask
        )
        attention_match_kd_loss = attention_match_kd_loss * head_num

        return attention_match_kd_loss

    def _compute_value_rel_loss(
        self,
        student_value_state_list: List[torch.Tensor],
        teacher_value_state_list: List[torch.Tensor],
        mask=None,
    ) -> torch.Tensor:
        student_last_value_state = student_value_state_list[-1]
        teacher_last_value_state = teacher_value_state_list[-1]
        # batch_size, head_num, seq_len, seq_len = student_last_value_state.size()
        # if mask is not None:
        #     mask = mask.unsqueeze(1).expand(-1, head_num, seq_len)
        #     mask = mask.reshape(-1, seq_len)
        value_rel_match_kd_loss = self.rel_loss_fct(
            student_last_value_state,
            student_last_value_state,
            teacher_last_value_state,
            teacher_last_value_state,
            mask,
        )
        return value_rel_match_kd_loss

    def _compute_loss_implement(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        input_mask: torch.Tensor,
    ):
        total_loss = None
        loss_dict = {}
        # compute kd loss
        for i, (student_logits, teacher_logits) in enumerate(
            zip(
                student_output["output_logits_list"],
                teacher_output["output_logits_list"],
            )
        ):
            kd_loss = self.kd_loss_fct(student_logits, teacher_logits, mask=input_mask)
            weight = 1.0
            if i < len(self.soft_target_inner_weight):
                weight = self.soft_target_inner_weight[i]
            if total_loss is None:
                total_loss = weight * kd_loss
            else:
                total_loss = total_loss + weight * kd_loss
        total_loss = self.soft_target_weight * total_loss
        loss_dict["kd_loss"] = kd_loss

        # compute hard label loss
        hard_label_loss = student_output.get("loss", None)
        if hard_label_loss is not None:
            total_loss = total_loss + self.hard_target_weight * hard_label_loss
            loss_dict["ori_loss"] = hard_label_loss
        # else:
        #     print("Error | hard_label_loss is none!")

        attention_loss = self._compute_attention_loss(
            student_output["attention_score_list"],
            teacher_output["attention_score_list"],
            input_mask,
        )
        value_rel_loss = self._compute_value_rel_loss(
            student_output["value_state_list"],
            teacher_output["value_state_list"],
            input_mask,
        )
        loss_dict["attention_loss"] = attention_loss
        loss_dict["value_rel_loss"] = value_rel_loss
        total_loss += attention_loss + value_rel_loss

        return (total_loss, loss_dict)

    def compute_loss(
        self,
        student_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        teacher_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        input_mask=None,
    ):
        # contruct teacher all output
        teacher_output_dict = {}
        teacher_output_dict["output_logits_list"] = [teacher_output_tuple_list.logits]
        teacher_intermediate_states = self.get_teacher_intermediate_states()
        teacher_output_dict.update(teacher_intermediate_states)
        # contruct student all output
        student_output_dict = {}
        student_output_dict["loss"] = student_output_tuple_list.loss
        student_output_dict["output_logits_list"] = [student_output_tuple_list.logits]
        student_intermediate_states = self.get_student_intermediate_states()
        student_output_dict.update(student_intermediate_states)
        return self._compute_loss_implement(
            student_output_dict, teacher_output_dict, input_mask
        )


class MinilmGeneralDistiller(MetaDistiller):
    def __init__(
        self,
        distill_config: Any,
        teacher_model_config: Any,
        student_model_config: Any,
        teacher_model: nn.Module,
        student_model: nn.Module,
    ):
        super(MinilmGeneralDistiller, self).__init__()
        self.clear_cache()
        is_init_from_teacher = (
            distill_config.is_init_from_teacher
            and (student_model_config.hidden_size == teacher_model_config.hidden_size)
            and (student_model_config.model_type == teacher_model_config.model_type)
        )
        if is_init_from_teacher:
            self.student_init_from_teacher(teacher_model.encoder, student_model.encoder)

        self.kd_loss_fct = KLDLoss(temperature=distill_config.temperature)
        self.attention_score_loss_fct = KLDLoss()
        self.rel_loss_fct = SelfAttentionRelationLoss()
        self.hard_target_weight = getattr(distill_config, "hard_target_weight", 0.001)
        self.soft_target_weight = getattr(distill_config, "soft_target_weight", 0.001)
        print("soft_target_weight: ", self.soft_target_weight)
        print("hard_target_weight: ", self.hard_target_weight)
        self.soft_target_inner_weight = getattr(
            distill_config, "soft_target_inner_weight", []
        )
        self.head_num = 64
        # self.pkd_weight = getattr(distill_config, "pkd_weight", 1.0)  # patient loss的权重

    def clear_cache(self) -> None:
        """
        重新初始化cache, 每次forward前执行，防止二次访问
        """
        self.teacher_intermediate_state = {
            "attention_layer_output_list": [],
        }  # 目前先只考虑单teacher
        self.student_intermediate_state = {
            "attention_layer_output_list": [],
        }  # cache the teacher and student intermediate_state
        return

    def register_student_module_hook(self, model: nn.Module) -> None:
        # TODO(@yutian) 修改为根据model type和特征名获取对应的register函数
        # register_bert_hidden_state_hook(
        #     model, self._hook_student_module_output, "value_state_list"
        # )
        # 这里优点疑问，value relation应该是以最初的value output算出来的，这里可以后面实验下
        register_bert_attention_and_value_state_hook(
            model, self._hook_student_module_output, "attention_layer_output_list"
        )
        return

    def register_teacher_module_hook(self, model: nn.Module) -> None:
        # register_bert_hidden_state_hook(
        #     model, self._hook_teacher_module_output, "value_state_list"
        # )
        register_bert_attention_and_value_state_hook(
            model, self._hook_teacher_module_output, "attention_layer_output_list"
        )
        return

    def student_init_from_teacher(
        self, teacher_model: nn.Module, student_model: nn.Module
    ) -> None:
        """
        定义了student如何从teacher继承参数，目前很简单，直接继承前x层
        """
        # teacher_state_dict = teacher_model.state_dict().copy()
        # # only load encoder param
        # load_weight_from_state_dict(student_model, teacher_state_dict)
        # return
        pass

    def get_teacher_intermediate_states(
        self,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        获取teacher的中间变量，pkd-bert是要拿到每一层的hidden_state, hidden_state对应输出层有多个输出， hidden_state位于第一个输出
        """
        res = {}
        # res["hidden_state_list"] = [
        #     x[0].detach() for x in self.teacher_intermediate_state["hidden_state_list"]
        # ]
        # res["attention_score_list"] = [
        #     x[-2].detach()
        #     for x in self.teacher_intermediate_state["attention_layer_output_list"]
        # ]
        res["query_state_list"] = [
            x[-3].detach()
            for x in self.teacher_intermediate_state["attention_layer_output_list"]
        ]
        res["key_state_list"] = [
            x[-2].detach()
            for x in self.teacher_intermediate_state["attention_layer_output_list"]
        ]
        res["value_state_list"] = [
            x[-1].detach()
            for x in self.teacher_intermediate_state["attention_layer_output_list"]
        ]
        # print(
        #     "inter state size",
        #     len(self.teacher_intermediate_state["attention_layer_output_list"]),
        # )
        # print("attention_score_list size", len(res["attention_score_list"]))
        # print("value_state_list size", len(res["value_state_list"]))

        return res

    def get_student_intermediate_states(self):
        """
        获取student的中间变量，pkd-bert是要拿到每一层的hidden_state, hidden_state对应输出层有多个输出， hidden_state位于第一个输出
        """
        res = {}
        # res["hidden_state_list"] = [
        #     x[0] for x in self.student_intermediate_state["hidden_state_list"]
        # ]
        res["query_state_list"] = [
            x[-3]
            for x in self.student_intermediate_state["attention_layer_output_list"]
        ]
        res["key_state_list"] = [
            x[-2]
            for x in self.student_intermediate_state["attention_layer_output_list"]
        ]
        res["value_state_list"] = [
            x[-1]
            for x in self.student_intermediate_state["attention_layer_output_list"]
        ]
        # print("student layer 11 val ", res["value_state_list"][-1].size())
        # print("student layer 11 val ", res["value_state_list"][-1][:2, :3, :3, :3])
        # print(
        #     "student layer 11 attn ",
        #     res["attention_score_list"][-1].size(),
        # )
        # print(
        #     "student layer 11 attn",
        #     res["attention_score_list"][-1][:2, :3, :3, :3],
        # )
        # print(
        #     "student state size",
        #     len(self.student_intermediate_state["attention_layer_output_list"]),
        # )
        # print("student attention_score_list size", len(res["attention_score_list"]))
        # print("student value_state_list size", len(res["value_state_list"]))
        return res

    def _split_to_same_head_num(self, state_tensor: torch.Tensor):
        batch_size, head_num, seq_len, head_size = state_tensor.size()
        state_tensor = state_tensor.permute(0, 2, 1, 3)
        state_tensor = state_tensor.reshape(batch_size, seq_len, self.head_num, -1)
        state_tensor = state_tensor.permute(0, 2, 1, 3)
        return state_tensor

    def _compute_attention_loss(
        self,
        student_query_state_list: List[torch.Tensor],
        student_key_state_list: List[torch.Tensor],
        teacher_query_state_list: List[torch.Tensor],
        teacher_key_state_list: List[torch.Tensor],
        mask=None,
    ) -> torch.Tensor:
        """
        计算q-k, 即attention scores对应的match loss, 当前只实现蒸馏最后一层
        Args:
        """
        student_last_query_state = self._split_to_same_head_num(
            student_query_state_list[-1]
        )
        teacher_last_query_state = self._split_to_same_head_num(
            teacher_query_state_list[-1]
        )
        student_last_key_state = self._split_to_same_head_num(
            student_key_state_list[-1]
        )
        teacher_last_key_state = self._split_to_same_head_num(
            teacher_key_state_list[-1]
        )
        attn_match_kd_loss = self.rel_loss_fct(
            student_last_query_state,
            student_last_key_state,
            teacher_last_query_state,
            teacher_last_key_state,
            mask,
        )
        return attn_match_kd_loss

    def _compute_value_rel_loss(
        self,
        student_value_state_list: List[torch.Tensor],
        teacher_value_state_list: List[torch.Tensor],
        mask=None,
    ) -> torch.Tensor:
        student_last_value_state = student_value_state_list[-1]
        teacher_last_value_state = teacher_value_state_list[-1]
        student_last_value_state = self._split_to_same_head_num(
            student_last_value_state
        )
        teacher_last_value_state = self._split_to_same_head_num(
            teacher_last_value_state
        )
        # if mask is not None:
        #     mask = mask.unsqueeze(1).expand(-1, head_num, seq_len)
        #     mask = mask.reshape(-1, seq_len)
        value_rel_match_kd_loss = self.rel_loss_fct(
            student_last_value_state,
            student_last_value_state,
            teacher_last_value_state,
            teacher_last_value_state,
            mask,
        )
        return value_rel_match_kd_loss

    def _compute_loss_implement(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        input_mask: torch.Tensor,
    ):
        total_loss = None
        loss_dict = {}
        # compute kd loss
        for i, (student_logits, teacher_logits) in enumerate(
            zip(
                student_output["output_logits_list"],
                teacher_output["output_logits_list"],
            )
        ):
            kd_loss = self.kd_loss_fct(student_logits, teacher_logits, mask=input_mask)
            weight = 1.0
            if i < len(self.soft_target_inner_weight):
                weight = self.soft_target_inner_weight[i]
            if total_loss is None:
                total_loss = weight * kd_loss
            else:
                total_loss = total_loss + weight * kd_loss
        total_loss = self.soft_target_weight * total_loss
        loss_dict["kd_loss"] = kd_loss

        # compute hard label loss
        hard_label_loss = student_output.get("loss", None)
        if hard_label_loss is not None:
            total_loss = total_loss + self.hard_target_weight * hard_label_loss
            loss_dict["ori_loss"] = hard_label_loss
        # else:
        #     print("Error | hard_label_loss is none!")

        attention_loss = self._compute_attention_loss(
            student_output["query_state_list"],
            student_output["key_state_list"],
            teacher_output["query_state_list"],
            teacher_output["key_state_list"],
            input_mask,
        )
        value_rel_loss = self._compute_value_rel_loss(
            student_output["value_state_list"],
            teacher_output["value_state_list"],
            input_mask,
        )
        loss_dict["attention_loss"] = attention_loss
        loss_dict["value_rel_loss"] = value_rel_loss
        total_loss = total_loss + attention_loss + value_rel_loss

        return (total_loss, loss_dict)

    def compute_loss(
        self,
        student_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        teacher_output_tuple_list: Tuple[torch.Tensor, Tuple[torch.Tensor]],
        input_mask=None,
    ):
        # contruct teacher all output
        teacher_output_dict = {}
        teacher_output_dict["output_logits_list"] = [teacher_output_tuple_list.logits]
        teacher_intermediate_states = self.get_teacher_intermediate_states()
        teacher_output_dict.update(teacher_intermediate_states)
        # contruct student all output
        student_output_dict = {}
        student_output_dict["loss"] = student_output_tuple_list.loss
        student_output_dict["output_logits_list"] = [student_output_tuple_list.logits]
        student_intermediate_states = self.get_student_intermediate_states()
        student_output_dict.update(student_intermediate_states)
        return self._compute_loss_implement(
            student_output_dict, teacher_output_dict, input_mask
        )


# if __name__ == "__main__":
# from models.base_modeling import BertModel
# from utils.config import ExpConfig
# from transformers import BertConfig
# from utils.text_utils import TextProcessor

# config = ExpConfig()
# setattr(config, "model_type", "bert")
# setattr(config, "dropout", 0.0)
# setattr(config, "num_labels", 2)
# model_config = BertConfig.from_json_file(
#     "./prev_trained_model/bert-base-chinese/config.json"
# )
# model = BertModel(model_config)

# for module_name, cur_module in model.named_modules():
#     print(module_name)

# distiller = PkdDistiller(
#     distill_config=None,
#     student_model_config=None,
#     teacher_model_config=None,
#     student_model=None,
#     teacher_model=None,
# )
# distiller.register_student_module_hook(model)

# fake_input = torch.Tensor([[1, 2, 3, 4, 5, 31, 58, 33, 12]]).long()
# output = model(fake_input, output_hidden_states=True, output_attentions=True)
# print("******raw_output*******")
# print(len(output.attentions))
# print(len(output.hidden_states))
# print(output.attentions[0].size())
# print(output.hidden_states[-1].size())
# print(output.hidden_states[8][0, 0, :10])

# print("******hook_output*******")
# res = distiller.get_student_intermediate_states()
# print(res["hidden_state_list"][-1].size())
# print(res["hidden_state_list"][7][0, 0, :10])
# # print(res["hidden_state_list"][-1])
