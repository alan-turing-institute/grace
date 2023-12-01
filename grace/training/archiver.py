import torch


class ModelArchiver:
    def __init__(self, model):
        self.model = model
        self.architecture = self.get_model_architecture(self.model)

    def get_model_architecture(self, module):
        architecture = {}
        for name, layer in module.named_children():
            layer_info = {}

            if isinstance(layer, torch.nn.ModuleList):
                layer_info[
                    str(layer.__class__.__name__)
                ] = self.get_model_architecture(layer)

            else:
                # Inputs:
                if hasattr(layer, "in_features"):
                    layer_info["input_shape"] = layer.in_features
                elif hasattr(layer, "in_channels"):
                    layer_info["input_shape"] = layer.in_channels
                else:
                    layer_info["input_shape"] = None

                # Outputs:
                if hasattr(layer, "out_features"):
                    layer_info["output_shape"] = layer.out_features
                elif hasattr(layer, "out_channels"):
                    layer_info["output_shape"] = layer.out_channels
                else:
                    layer_info["output_shape"] = None

            architecture[name] = layer_info
        return architecture
