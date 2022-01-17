import hydra
import matplotlib.pyplot as plt
import sklearn.manifold as mani
import torch
import torchdrift
from model import MyAwesomeModel

from src.data.dataset import MnistDataset

# Training params
batch_size = 0
learning_rate = 0
epochs = 0
data_dir = ""

# Model params
seed = 0


@hydra.main(config_name="training_config.yaml")
def set_training_params(cfg):
    global learning_rate
    learning_rate = cfg.hyperparameters.learning_rate
    global batch_size
    batch_size = cfg.hyperparameters.batch_size
    global epochs
    epochs = cfg.hyperparameters.epochs
    global data_dir
    data_dir = cfg.hyperparameters.data_dir


@hydra.main(config_name="model_config.yaml")
def set_model_params(cfg):
    global seed
    seed = cfg.hyperparameters.seed


def main():
    print(f'Training data dir: {data_dir}')

    model = MyAwesomeModel()

    train_set = MnistDataset(data_dir)
    print(len(train_set))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.load_state_dict(torch.load('models/running_model.pth'))
    model.double()

    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(trainloader, model, drift_detector)
    # drift_detection_model = torch.nn.Sequential(
    #     model,
    #     drift_detector
    # )
    inputs, labels = next(iter(trainloader))
    features = model(inputs)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print(score), print(p_val)

    # N_base = drift_detector.base_outputs.size(0)
    mapper = mani.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features.detach().numpy())
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f'score {score:.2f} p-value {p_val:.2f}')
    plt.show()

    # log_ps = model(images.float())
    # loss = criterion(log_ps, labels)
    # loss.backward()
    # optimizer.step()

    # plt.plot(train_losses)
    # plt.xlabel("Epochs")
    # plt.ylabel("Training loss")
    # os.makedirs("reports/figures/", exist_ok=True)
    # plt.savefig("reports/figures/training_plot.png")


if __name__ == '__main__':
    set_training_params()
    set_model_params()
    main()
