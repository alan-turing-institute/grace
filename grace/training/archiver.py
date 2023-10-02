class ModelArchiver:
    def __init__(self, model):
        self.model = model
        self.architecture = self.get_model_architecture()

    def get_model_architecture(self):
        architecture = {}
        for name, layer in self.model.named_children():
            layer_info = {
                "type": str(layer.__class__.__name__),
                "input_size": layer.in_features
                if hasattr(layer, "in_features")
                else None,
                "output_size": layer.out_features
                if hasattr(layer, "out_features")
                else None,
            }
            architecture[name] = layer_info
        return architecture
