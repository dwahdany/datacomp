import mlflow
from datacomp.encoders import CLIPTextEncoder
from datacomp.tokenizers import CLIPTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def zero_shot_eval(model, dataset, text_encoder):
    """
    Perform zero-shot evaluation on the specified dataset.

    Args:
        model: The trained model to evaluate.
        dataset (str): Name of the dataset ('imagenet', 'pcam', or 'cifar100').
        text_encoder: Function to encode text prompts.

    Returns:
        float: Accuracy of the zero-shot evaluation.
    """
    if dataset == "imagenet":
        eval_dataset = datasets.ImageNet(
            root="./data", split="val", transform=transforms.ToTensor()
        )
    elif dataset == "pcam":
        # TODO: Implement PCAM dataset loading
        raise NotImplementedError("PCAM dataset loading not implemented yet")
    elif dataset == "cifar100":
        eval_dataset = datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    # Prepare class names and text prompts
    class_names = eval_dataset.classes
    text_prompts = [f"a photo of a {name}" for name in class_names]
    encoded_text = text_encoder(text_prompts)

    all_predictions = []
    all_labels = []

    for images, labels in dataloader:
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = encoded_text / encoded_text.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predictions = similarity.argmax(dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy


def evaluate_model(model, text_encoder):
    """
    Evaluate the model on ImageNet, PCAM, and CIFAR100.

    Args:
        model: The trained model to evaluate.
        text_encoder: Function to encode text prompts.
    """
    mlflow.start_run()

    datasets = ["imagenet", "pcam", "cifar100"]
    for dataset in datasets:
        accuracy = zero_shot_eval(model, dataset, text_encoder)
        mlflow.log_metric(f"{dataset}_accuracy", accuracy)

    mlflow.end_run()


if __name__ == "__main__":
    # Example usage (you would need to load your actual model and text encoder)
    model = None  # Placeholder for your trained model
    tokenizer = CLIPTokenizer()
    text_encoder = CLIPTextEncoder(tokenizer)
    evaluate_model(model, text_encoder)
