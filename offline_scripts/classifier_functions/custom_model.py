from huggingface_hub import hf_hub_download
from torch import nn


class CustomModel(nn.Module):
    def __init__(self, input_size, perceptron_size, pretrained_model, output_nn_size):
        super(CustomModel, self).__init__()

        self.perceptron = nn.Linear(input_size, perceptron_size)
        self.pretrained = pretrained_model

        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.pretrained.fc = nn.Linear(self.pretrained.fc.in_features, output_nn_size)
        self.pretrained.fc.requires_grad = True

    def forward(self, x):
        x = self.perceptron(x)
        x = self.pretrained(x)
        return x

if __name__ == '__main__':
    import pickle

    # download the model from the hub:
    path_kwargs = hf_hub_download(
        repo_id='PierreGtch/EEGNetv4',
        filename='EEGNetv4_Lee2019_MI/kwargs.pkl',
    )
    path_params = hf_hub_download(
        repo_id='PierreGtch/EEGNetv4',
        filename='EEGNetv4_Lee2019_MI/model-params.pkl',
    )
    with open(path_kwargs, 'rb') as f:
        kwargs = pickle.load(f)
    module_cls = kwargs['module_cls']
    module_kwargs = kwargs['module_kwargs']

    # load the model with pre-trained weights:
    torch_module = module_cls(**module_kwargs)