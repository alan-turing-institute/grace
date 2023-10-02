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


#     # def save_to_json(self, filename):
#     #     with open(filename, "w") as json_file:
#     #         json.dump(self.architecture, json_file, indent=4)

# # Example usage:
# # Create and initialize your custom model
# model_archiver = ModelArchiver(model)

# # Save the model architecture to a JSON file
# model_archiver.save_to_json("model_architecture.json")
