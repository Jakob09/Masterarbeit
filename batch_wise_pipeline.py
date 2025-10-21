from torchvision.models import resnet50, ResNet50_Weights
import os
import sys
import gc
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, KPCA_CAM
from pytorch_grad_cam import ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import foolbox as fb
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity
import foolbox.attacks as fb_att
from foolbox.distances import LpDistance
from scipy.special import kl_div
from datasets import load_dataset
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

''' function to denormalize a numpy image
arguments: image_np: NumPy array of shape (H, W, C), float32, normalized
returns: unnormalized image in [0, 1] range'''
def denormalize_np(image_np):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    image_np = image_np.astype(np.float32)
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return np.clip((image_np * std + mean), 0, 1)

''' function to make a class prediction for a batch of already preprocessed images
arguments: model: resnet50 model, img_batch: tensor of preprocessed images
returns: List of (category_name, score) tuples per image (category_name: string of predicted category, score: score of prediction)'''
def make_batch_predictions(model, img_batch):
    weights = ResNet50_Weights.DEFAULT
    with torch.no_grad():
        outputs = model(img_batch)
        probs = torch.softmax(outputs, dim=1)
        top_scores, top_classes = probs.max(dim=1)
        results = []
        for class_id, score in zip(top_classes, top_scores):
            category_name = weights.meta["categories"][class_id.item()]
            results.append((category_name, class_id.item(), score.item()))
    return results


''' function to compute the explanation for the predicted classes
arguments: img_tensor: tensor of already preprocessed images, target_layers: model layers to get explanation from
targets: list of classes targeted , model: resnet50 model, method: explanation method
returns: grayscale_cam: numpy array with pixel importances for each image in the batch'''
def calculate_multiple_explanations(batch, target_layers, targets, model, method):
    # === Compute CAM method ===
        # Set aug_smooth=True and eigen_smooth=True for smoother maps (optional)
    with method(model=model, target_layers=target_layers) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=batch,
                            targets=targets)
                            #aug_smooth=True,
                            #eigen_smooth=True)
        return grayscale_cam

''' function to visualize the image with pixel highlight overlay of explanation method
arguments: image_np: float32 with values in [0,1] as numpy array with (H, W, C) format
grayscale_cam: output produced by explanation method
prediction: string representing predicted class 
method: xAI method used to produce cam'''
def visualize_explanation(image_np, grayscale_cam, prediction, method):
    image_np = denormalize_np(image_np)
    method_name = str(method).split(".")[-1].split("\'")[0]
    visualization = show_cam_on_image(denormalize_np(image_np), grayscale_cam, use_rgb=True)

    # === Display using Matplotlib ===
    fig_vis, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(denormalize_np(image_np)) 
    axes[0].set_title("Original Image\n")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title(f"{method_name}\nTarget/Pred: {prediction}")
    axes[1].axis('off')

    fig_vis.suptitle(f"{method_name} for ResNet50")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # print(f"{method_name} visualization complete.")
    return fig_vis

''' function to combine visualization of the original image and adversarial image and the corresponding saliency maps'''
def compare_visualizations(orig_image_np, grayscale_cam_orig, pred_orig, adv_image_np, grayscale_cam_adv, pred_adv, method_name, attack_name):
    orig_image_np = denormalize_np(orig_image_np)                                       # convert to [0, 1] range for visualization
    adv_image_np = denormalize_np(adv_image_np)

    orig_visualization = show_cam_on_image(orig_image_np, grayscale_cam_orig, use_rgb=True)
    adv_visualization = show_cam_on_image(adv_image_np, grayscale_cam_adv, use_rgb=True)

    # === Display using Matplotlib ===
    fig, axes = plt.subplots(2, 2)  # Slightly smaller width

    axes[0, 0].imshow(orig_image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(orig_visualization)
    axes[0, 1].set_title(f"{method_name}\nPredicted class: {pred_orig}")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(adv_image_np)
    axes[1, 0].set_title("Adversarial Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(adv_visualization)
    axes[1, 1].set_title(f"{method_name}\nPredicted class: {pred_adv}")
    axes[1, 1].axis('off')

    fig.suptitle(f"Attack: {attack_name}")

    # Precise control of spacing
    fig.subplots_adjust(wspace=0.1, hspace=0.4, top=0.84)

    return fig


''' function to calculate the difference in explanation for original and adversarial image
arguments: grayscale_cam_orig: pixel importances for original image, 
grayscale_cam_adv: pixel importances for adversarial input
returns: mean of pixelwise differences'''
def mean_pixel_difference(grayscale_cam_orig, grayscale_cam_adv):
    return np.mean(np.abs(grayscale_cam_orig - grayscale_cam_adv))

''' function to count the number of pixels that are significantly different for both explanations
arguments: grayscale_cam_orig: pixel importances for original image, 
grayscale_cam_adv: pixel importances for adversarial input
returns: number of pixels, where the importance is significantly different (difference greater than 0.2)'''
def pixel_differrences_count(grayscale_cam_orig, grayscale_cam_adv):
    differences = np.abs(grayscale_cam_orig - grayscale_cam_adv)
    return np.count_nonzero(differences > 0.2)

''' function to measure how the number of important pixels changes for the original and adversarial image based on a threshold
arguments: grayscale_cam_orig: pixel importances for original image, 
grayscale_cam_adv: pixel importances for adversarial input
threshold: pixel importance threshold to consider a pixel as activated
returns: ratio of number of activated pixels in adversarial explanation to original explanation'''
def pixel_activation_ratio(grayscale_cam_orig, grayscale_cam_adv, threshold=0):
    pixels_activated_orig = np.count_nonzero(grayscale_cam_orig > threshold)
    pixels_activated_adv = np.count_nonzero(grayscale_cam_adv > threshold)
    return pixels_activated_adv/pixels_activated_orig

''' function to count the number of regions (connected components) that are considered highly relevant 
(pixel importance higher than 0.9)
arguments: saliency_map: numpy array of pixel importances
    threshold: pixel importance threshold to consider a pixel as highly relevant
returns: number of connected components in the binary map of highly relevant pixels'''
def count_attention_regions(saliency_map: np.ndarray, threshold: float = 0.9) -> int:
    binary_map = (saliency_map >= threshold).astype(np.uint8)
    # Label connected components (8-connectivity)
    labeled_map, num_regions = scipy.ndimage.label(binary_map, structure=np.ones((3, 3)))
    return num_regions

''' function to turn the saliency map into a probability distribution for calculation of Kullback-Leibler Divergence
arguments: saliency_map: numpy array of pixel importances
returns: normalized saliency map, adding small epsilon to avoid 0 values'''
def normalize_saliency_map(saliency_map: np.ndarray):
    saliency_map = saliency_map + 1e-10
    return saliency_map / np.sum(saliency_map)

''' function to compute the Jenson-Shannon Divergence (symmetric distance for probability distributions)
arguments: saliency1: numpy array of pixel importances for original image,
           saliency2: numpy array of pixel importances for adversarial image
returns: JS Divergence value'''
def compute_js_divergence(saliency1: np.ndarray, saliency2: np.ndarray):
    saliency1_norm = normalize_saliency_map(saliency1)
    saliency2_norm = normalize_saliency_map(saliency2)

    # Compute M = (P + Q) / 2
    M = (saliency1_norm + saliency2_norm) / 2

    # Compute KL(P || M) and KL(Q || M)
    kl_pm = np.sum(kl_div(saliency1_norm, M))
    kl_qm = np.sum(kl_div(saliency2_norm, M))

    # JS Divergence
    js_divergence = 0.5 * (kl_pm + kl_qm)
    return js_divergence

''' function to compute the intersection over union for the saliency maps of original and adversarial image
arguments: saliency_map_orig: pixel importances of original image,
           saliency_map_adv: pixel importances of adversarial
           threshold_a: quantile threshold for original saliency map to consider pixel as relevant
           threshold_b: quantile threshold for adversarial saliency map to consider pixel as relevant
returns: IoU value'''
def compute_iou(saliency_map_orig, saliency_map_adv, threshold_a=0.8, threshold_b=0.8):
    A_mask = saliency_map_orig > np.quantile(saliency_map_orig, threshold_a)
    B_mask = saliency_map_adv > np.quantile(saliency_map_adv, threshold_b)

    # calculate IoU
    intersection = np.logical_and(A_mask, B_mask).sum()
    union = np.logical_or(A_mask, B_mask).sum()
    return intersection / union if union > 0 else 0.0

''' function to calculate Spearman's rank correlation coefficient between two saliency maps
arguments: saliency_map_orig: pixel importances of original image,
           saliency_map_adv: pixel importances of adversarial
returns: Spearman's rank correlation coefficient'''
def calculate_spearman_rank_correlation(saliency_map_orig, saliency_map_adv):
    original_flat = saliency_map_orig.flatten()
    adversarial_flat = saliency_map_adv.flatten()
    correlation_coefficient, p_value = scipy.stats.spearmanr(original_flat, adversarial_flat)
    return correlation_coefficient

''' function to calculate metrics to compare the original saliency map and the one for the adversarial.
arguments: grayscale_cam_orig: pixel importances of original image,
           grayscale_cam_adv: pixel importances of adversarial
returns: values of all relevant metrics'''
def calculate_metrics(grayscale_cam_orig, grayscale_cam_adv):
    mean_pixel_diff = mean_pixel_difference(grayscale_cam_orig, grayscale_cam_adv)
    percent_different_pixels = pixel_differrences_count(grayscale_cam_orig, grayscale_cam_adv) / (grayscale_cam_orig.shape[0] * grayscale_cam_orig.shape[1]) * 100
    cosine_sim = cosine_similarity(grayscale_cam_orig.flatten().reshape(1, -1), grayscale_cam_adv.flatten().reshape(1, -1))[0, 0]
    activation_ratio = pixel_activation_ratio(grayscale_cam_orig, grayscale_cam_adv)
    js_div = compute_js_divergence(grayscale_cam_orig, grayscale_cam_adv)
    highly_relevant_ratio = pixel_activation_ratio(grayscale_cam_orig, grayscale_cam_adv, threshold=0.7)
    intersection_over_union = compute_iou(grayscale_cam_orig, grayscale_cam_adv)

    print(f"The mean absolute difference in pixel importances is {(mean_pixel_diff):.4f}")
    print(f"The percentage of pixels that are significantly different is {(percent_different_pixels):.2f}%")
    print(f"The cosine similarity is {(cosine_sim):.3f}")
    print(f"The ratio of activated pixels is: {(activation_ratio):.3f}")
    print(f"The ratio of highly relevant pixels is: {(highly_relevant_ratio):.3f}")
    print(f"JS Divergence: {js_div}")

    num_regions_orig = count_attention_regions(grayscale_cam_orig, threshold=0.8)
    num_regions_adv = count_attention_regions(grayscale_cam_adv, threshold=0.8)
    print(f"Number of attention regions original: {num_regions_orig}")
    print(f"Number of attention regions adversarial: {num_regions_adv}")
    print(f"The intersection over union is: {(intersection_over_union):.3f}")
    print("\n")

    return mean_pixel_diff, percent_different_pixels, cosine_sim, activation_ratio, highly_relevant_ratio, js_div, num_regions_orig, num_regions_adv, intersection_over_union

''' function to create adversarial images for a batch of images using the specified attack and epsilons
arguments: batch of preprocessed images (Tensor of size [N, 3, 224, 224]), label (Tensor of size [N] containing the class_ids),
attack: Class from foolbox.attacks, epsilons: numpy array of epsilons to try for that attack
returns: selected_advs: adversarial images (Tensor of size [N, 3, 224, 224]) (None if no adversarial was produced)'''
def create_untargeted_adversarials(image_batch, label_batch, attack, epsilons):

    raw, clipped, is_adv = attack(fmodel, image_batch, label_batch, epsilons=epsilons)                      # clipped: list of tensors of size [N, 3, 224, 224] -> index of list corresponds to epsilon
                                                                                                            # different images might have different best epsilons
    selected_advs = []
    K, N = is_adv.shape                         # K = #epsilons, N = #images

    for i in range(N):                  # for each image in batch
        success = is_adv[:, i]
        indices = torch.nonzero(success, as_tuple=False)
        if indices.numel() > 0:
            first_true_index = indices[0][0].item()                                             # smallest epsilon that succeeded
            adv_image = clipped[first_true_index][i]
            selected_advs.append(adv_image)
            print(f"Adversarial produced with Epsilon: {epsilons[first_true_index]}")
        else:
            print("No adversarial was produced.")
            selected_advs.append(None)

    return selected_advs

''' function to run the explanation on the original image and compare the explanation for the adversarial with the original one
saves generated plots (comparison of heatmaps) and stores metrics of comparison in csv file
arguments: model: resnet-50 model, images: preprocessed images (Tensor of size [N, 3, 224, 224]), labels: (Tensor of size [N] containing class_id),
xAImethod: class from pytorch-gradcam, attack: class from foolbox.attacks, adversarials: Tensor of size [N, 3, 224, 224],
csv_file: path to csv file to store metrics, ids: Tensor of size [N] containing image ids'''
def create_and_compare_explanations(model, images, labels, xAImethod, attack, adversarials, csv_file, ids):
    predictions = make_batch_predictions(model, images)
    weights = ResNet50_Weights.DEFAULT

    target_layers = [model.layer4[-1]]
    targets = None

    adv_predictions = make_batch_predictions(model, adversarials)

    batch = torch.cat([images, adversarials], dim=0)

    grayscale_cams = calculate_multiple_explanations(batch, target_layers, targets, model, xAImethod)

    number_of_images = images.shape[0]

    for i in range(number_of_images):
        image = images[i]
        adv_image = adversarials[i]
        id = ids[i].item()
        orig_pred_str = predictions[i][0]
        score = predictions[i][2]
        adv_pred_str = adv_predictions[i][0]
        score_adv = adv_predictions[i][2]
        print("ORIGINAL PREDICTION: ", orig_pred_str, " with score: ", score)
        print("ADVERSARIAL PREDICTION: ", adv_pred_str, " with score: ", score_adv)

        image_np = image.permute(1, 2, 0).cpu().numpy()
        adv_image_np = adv_image.permute(1, 2, 0).cpu().numpy()
        grayscale_cam_orig = grayscale_cams[i, :]
        grayscale_cam_adv = grayscale_cams[(i + number_of_images), :]

        method_name = str(xAImethod).split(".")[-1].split("\'")[0]
        attack_name = str(attack).split("(")[0]
        class_name = weights.meta["categories"][labels[i]]

        images_similarity = torch.mean((image - adv_image)**2).item()
        print(f"Difference of images: {images_similarity}")

        ### saving metrics
        mean_pixel_diff, percent_different_pixels, cosine_sim, activation_ratio, highly_relevant_ratio, js_div, num_regions_orig, num_regions_adv, intersection_over_union = calculate_metrics(grayscale_cam_orig, grayscale_cam_adv)

        ssim_value = structural_similarity(grayscale_cam_orig, grayscale_cam_adv, data_range=1)
        spearman_rank_coeff = calculate_spearman_rank_correlation(grayscale_cam_orig, grayscale_cam_adv)

        # Write header if the file does not exist
        write_header = not os.path.exists(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow([
                    "image_id", "method_name", "attack_name", "class_name", "target_pred", "images_mean_difference", "score_original", "score_adversarial",
                    "mean_pixel_diff", "percent_different_pixels", "num_regions_orig", "num_regions_adv",
                    "cosine_sim", "activation_ratio", "js_div", "highly_relevant_pixels_ratio", "intersection_over_union", "ssim", "spearman_rank_coeff"
                ])
            writer.writerow([
                id, method_name, attack_name, class_name, adv_pred_str, images_similarity, score, score_adv,
                mean_pixel_diff, percent_different_pixels, num_regions_orig, num_regions_adv, cosine_sim, 
                activation_ratio, js_div, highly_relevant_ratio, intersection_over_union, ssim_value, spearman_rank_coeff
            ])
    del grayscale_cam_adv, grayscale_cam_orig, grayscale_cams, batch


''' function to load images from imagenet subset and preprocess them
arguments: preprocess: preprocessing function from torchvision models, dataset_url: string to specify dataset
returns: tensor of preprocessed images and tensor of labels (class_ids)'''
def load_and_transform_images(preprocess, dataset_url="ioxil/imagenetsubset"):
    images = []
    labels = []

    if dataset_url == "ioxil/imagenetsubset":
        dataset = load_dataset("ioxil/imagenetsubset")
        test_data = dataset["train"]
    else:
        dataset = load_dataset("Multimodal-Fatima/Imagenet1k_sample_validation")
        test_data = dataset["validation"]

    for sample in test_data:
        image = sample["image"].convert("RGB")
        class_id = sample["label"]
        tensor_image = preprocess(image)
        images.append(tensor_image)
        labels.append(torch.tensor(class_id))

    return torch.stack(images), torch.tensor(labels)

'''function to create result folders if they do not exist yet'''
def setup_result_folders():
    # Setup Folders
    results_folder = 'results'
    images_folder = os.path.join(results_folder, 'images')
    targeted_folder = os.path.join(results_folder, "targeted")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(targeted_folder, exist_ok=True)
    print(f"Results will be saved in: {results_folder}")

'''function to filter out images that were not correctly classified by the model
arguments: img_batch: tensor of preprocessed images, label_batch: tensor of class_ids,
ids: tensor of image ids, predictions: list of (category_name, class_id, score) tuples
returns: filtered img_batch, label_batch, ids'''
def filter_wrong_classifications(img_batch, label_batch, ids, predictions):
    preds_class_id = torch.tensor([pred[1] for pred in predictions], device=label_batch.device)
    correct_mask = preds_class_id == label_batch
    img_batch = img_batch[correct_mask]
    label_batch = label_batch[correct_mask]
    ids = ids[correct_mask]
    return img_batch, label_batch, ids
    
'''function to filter out images where no adversarial could be created
arguments: img_batch: tensor of preprocessed images, label_batch: tensor of class_ids,
ids: tensor of image ids, adversarials: list of adversarial images
returns: filtered img_batch, label_batch, ids, adversarials'''
def filter_adversarial_fails(img_batch, label_batch, ids, adversarials):
    valid_indices = [i for i, adv in enumerate(adversarials) if adv is not None]
    img_batch = img_batch[valid_indices]
    label_batch = label_batch[valid_indices]
    ids = ids[valid_indices]
    adversarials = [adversarials[i] for i in valid_indices]
    if adversarials:
        adversarials = torch.stack(adversarials)
    else:
        adversarials = None
    return img_batch, label_batch, ids, adversarials

''' function to process a batch in micro-batches to avoid memory issues with certain explanation methods
arguments: model: resnet-50 model, batch_images: tensor of preprocessed images (Tensor of size [N, 3, 224, 224]), batch_labels: (Tensor of size [N] containing class_id),
xAImethod: class from pytorch-gradcam, attack: class from foolbox.attacks, adv_images: Tensor of size [N, 3, 224, 224],
csv_file: path to csv file to store metrics, batch_ids: Tensor of size [N] containing image ids,
micro_batch_size: size of micro-batches to split the batch into, device: torch device (cpu or cuda)'''
def create_explanations_micro_batch_wise(model, batch_images, batch_labels, xAImethod, attack, adv_images, csv_file, batch_ids, micro_batch_size, device):
    for start in range(0, batch_images.shape[0], micro_batch_size):
                            end = start + micro_batch_size
                            create_and_compare_explanations(model, batch_images[start:end], batch_labels[start:end], xAImethod, attack, adv_images[start:end], csv_file, batch_ids[start:end])
                            gc.collect()
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()


if __name__ == '__main__':
    # Check if argument is provided
    if len(sys.argv) < 2:
        print("Error: Missing argument. Usage: python batch_wise_pipeline.py <index>")
        sys.exit(1)

    # Get the argument (e.g., 0, 1, 2, or 3)
    attack_group_index = int(sys.argv[1])
    print(f"Running pipeline with index: {attack_group_index}")
    setup_result_folders()


    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Using pretrained weights and model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).eval()
    model = model.to(device)

    preprocess = weights.transforms()
    bounds = (-2.4, 2.8)
    
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)

    images, labels = load_and_transform_images(preprocess, dataset_url="Multimodal-Fatima/Imagenet1k_sample_validation")

    ids = torch.arange(len(images))

    dataset = TensorDataset(images, labels, ids)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

                   
    attack_to_epsilon = [ 
        {fb_att.LinfFastGradientAttack(): np.linspace(0, 1, num=20)},
        {fb_att.LinfProjectedGradientDescentAttack(): np.linspace(0, 0.05, num=5)},
        {fb_att.L2FastGradientAttack(): np.linspace(1, 150, num=20)},
        {fb_att.L2ProjectedGradientDescentAttack(): np.linspace(0.5, 10, num=5)},
        {fb_att.LInfFMNAttack(): np.linspace(0, 0.5, num=20)},
        {fb_att.L2FMNAttack(): np.linspace(0, 10, num=20)},
        {fb_att.L1FMNAttack(): np.linspace(2, 150, num=20)},
        {fb_att.LinfinityBrendelBethgeAttack(steps=100): np.linspace(0, 0.5, num=10)},
        {fb_att.L2RepeatedAdditiveUniformNoiseAttack(): np.linspace(1, 250, num=25)},
        {fb_att.L2RepeatedAdditiveGaussianNoiseAttack(): np.linspace(1, 250, num=25)},
        {fb_att.L2BrendelBethgeAttack(steps=100): np.linspace(0, 5, num=20)},
        {fb_att.LinearSearchBlendedUniformNoiseAttack(distance=LpDistance(100)): np.linspace(0, 20, num=20)},
        {fb_att.SaltAndPepperNoiseAttack(): np.linspace(1, 250, num=20)},
        {fb_att.LinfDeepFoolAttack(): np.linspace(0, 0.5, num=10)},
        {fb_att.L2DeepFoolAttack(): np.linspace(0, 10, num=20)},
        {fb_att.GaussianBlurAttack(distance=LpDistance(2)): np.linspace(1, 200, num=20)},
        {fb_att.L2ClippingAwareAdditiveUniformNoiseAttack(): np.linspace(1, 250, num=20)},
        {fb_att.LinfRepeatedAdditiveUniformNoiseAttack(): np.linspace(0.1, 3, num=20)},
        {fb_att.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(): np.linspace(1, 250, num=25)},
        {fb_att.L1BrendelBethgeAttack(steps=100): np.linspace(2, 100, num=10)},
        {fb_att.LinfBasicIterativeAttack(): np.linspace(0, 0.1, num=15)},
        {fb_att.BoundaryAttack(steps=10000): np.linspace(1, 150, num=10)},
        {fb_att.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(): np.linspace(1, 250, num=25)},
        {fb_att.LinfAdditiveUniformNoiseAttack(): np.linspace(0.1, 3, num=20)},
        {fb_att.L2ClippingAwareAdditiveGaussianNoiseAttack(): np.linspace(1, 250, num=20)},
        {fb_att.L2AdditiveUniformNoiseAttack(): np.linspace(1, 250, num=20)},
        {fb_att.VirtualAdversarialAttack(steps=100): np.linspace(10, 150, num=50)},
        {fb_att.DDNAttack(): np.linspace(0, 10, num=20)},
        {fb_att.L2BasicIterativeAttack(): np.linspace(0, 15, num=30)},
        {fb_att.EADAttack(steps=4000): np.linspace(1, 200, num=15)},
        {fb_att.L2AdditiveGaussianNoiseAttack(): np.linspace(1, 250, num=20)},
        {fb_att.NewtonFoolAttack(): np.linspace(0, 50, num=20)},
        {fb_att.L2CarliniWagnerAttack(steps=1000): np.linspace(0, 10, num=10)} 
        ]

    all_attacks = attack_to_epsilon[attack_group_index]

    # Define the path to the CSV file
    csv_file = "results/cam_comparison_metrics" + str(attack_group_index) + ".csv"

    explanation_methods = {"GradCAM": GradCAM,  "GradCAMPlusPlus": GradCAMPlusPlus, "EigenCAM": EigenCAM, 
                           "EigenGradCAM": EigenGradCAM, "LayerCAM": LayerCAM, "KPCA_CAM": KPCA_CAM, 
                           "AblationCAM": AblationCAM, "FullGrad": FullGrad, "ScoreCAM": ScoreCAM}
    
    # give exact same saliency maps: [GradCAM, HiResCAM, XGradCAM], [LayerCAM, GradCAMElementwise]

    for attack, epsilons in all_attacks.items():
        num_adv_failed = 0
        print(f"Doing attack: {attack}")
        ## create adversarial images then use explanation methods
        for batch_images, batch_labels, batch_ids in data_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_ids = batch_ids.to(device)

            predictions = make_batch_predictions(model, batch_images)

            ## check if predictions are correct: 
            batch_images, batch_labels, batch_ids = filter_wrong_classifications(batch_images, batch_labels, batch_ids, predictions)

            ## create adversarials for the batch
            adv_images = create_untargeted_adversarials(batch_images, batch_labels, attack, epsilons)
            num_none = sum(adv is None for adv in adv_images)
            print(f"No adversarial in {num_none} cases for current batch.")
            num_adv_failed += num_none

            batch_images, batch_labels, batch_ids, adv_images = filter_adversarial_fails(batch_images, batch_labels, batch_ids, adv_images)
            if adv_images is None:
                continue

            for xAImethod_name, xAImethod in explanation_methods.items():
                print(f"Trying explanation method: {xAImethod}")
                if xAImethod_name == "FullGrad":
                    micro_batch_size = 8
                    create_explanations_micro_batch_wise(model, batch_images, batch_labels, xAImethod, attack, adv_images, csv_file, batch_ids, micro_batch_size, device)
                elif xAImethod_name == "ScoreCAM":
                    micro_batch_size = 1
                    create_explanations_micro_batch_wise(model, batch_images, batch_labels, xAImethod, attack, adv_images, csv_file, batch_ids, micro_batch_size, device)
                else:
                    micro_batch_size = 64
                    create_explanations_micro_batch_wise(model, batch_images, batch_labels, xAImethod, attack, adv_images, csv_file, batch_ids, micro_batch_size, device)
            torch.cuda.empty_cache()
        
        print(f"Failed to create adversarial in {num_adv_failed} cases. For attack: {attack}. \n")
            


