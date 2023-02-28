import re


def register_bert_attention_and_value_state_hook(module, hook_fct, cache_key):
    # module_name must be xxx.layer.num.attention.self, e.g. encoder.layer.2
    self_attn_layer_list = []
    for module_name, cur_module in module.named_modules():
        re_output = re.search(".*layer\.\d+\.attention\.self$", module_name)
        if re_output is None:
            # 没有匹配的attention层，skip
            continue
        self_attn_layer_list.append(module_name)
    last_module_name = self_attn_layer_list[-1]
    for module_name, cur_module in module.named_modules():
        if module_name == last_module_name:
            cur_module.register_forward_hook(hook_fct(cache_key))
            setattr(cur_module, "output_attentions", True)
            print("register {} output hook".format(last_module_name))
    return


def register_bert_hidden_state_hook(module, hook_fct, cache_key):
    # module_name must be xxx.layer.num.xxx, e.g. encoder.layer.2
    for module_name, cur_module in module.named_modules():
        re_output = re.search(".*layer\.\d+$", module_name)
        if re_output is None:
            # 没有匹配的hidden层，skip
            continue
        cur_module.register_forward_hook(hook_fct(cache_key))
        print("register {} output hook".format(module_name))
    return
