from torchvision.models import resnet50, ResNet50_Weights
import os
import sys
import gc
import csv
import torch
import numpy as np
import torchvision.io as io
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, KPCA_CAM
from pytorch_grad_cam import ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import foolbox as fb
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity
import foolbox.attacks as fb_att
from foolbox.distances import LpDistance
from scipy.special import kl_div
from datasets import load_dataset # type: ignore

"""
tensor: Tensor of shape [C, H, W] normalized with given mean and std
mean/std: sequences of 3 values for RGB channels
"""
def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

"""
image_np: NumPy array of shape (H, W, C), float32, normalized
mean, std: lists of 3 floats
Returns: unnormalized image in [0, 1] range
"""
def denormalize_np(image_np):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    image_np = image_np.astype(np.float32)
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return np.clip((image_np * std + mean), 0, 1)

''' function to get the targets and target_layers needed for the gradCam explanation
arguments: model: resnet50 model, prediction: string representation of predicted class
returns: target_layers: model layers to get explanation from, targets: class targeted'''
def get_targets(model, prediction):
    weights = ResNet50_Weights.DEFAULT
    class_id = weights.meta["categories"].index(prediction)
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(class_id)]
    return targets, target_layers

''' function to make a class prediction for a already preprocessed image
arguments: model: resnet50 model, img: tensor of preprocessed image
returns: category_name: string of predicted category, score of prediction'''
def make_prediction(model, img):
    weights = ResNet50_Weights.DEFAULT
    with torch.no_grad():
        batch = img.unsqueeze(0)
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")
    return category_name, score

''' function to compute the explanation for the predicted class
arguments: img_tensor: already preprocessed images, target_layers: model layers to get explanation from
targets: class targeted , model: resnet50 model, method: explanation method
returns: grayscale_cam: explanation with highlighted pixels'''
def calculate_explanation(img_tensor, target_layers, targets, model, method):
    batch = img_tensor.unsqueeze(0)
    # === Compute Grad-CAM ===
        # Set aug_smooth=True and eigen_smooth=True for smoother maps (optional)
    with method(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=batch,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :] # Get the CAM for the first (and only) image

        return grayscale_cam

''' function to compute the explanation for the predicted classes
arguments: img_tensor: already preprocessed images, target_layers: model layers to get explanation from
targets: list of classes targeted , model: resnet50 model, method: explanation method
returns: grayscale_cam: explanations with highlighted pixels'''
def calculate_multiple_explanations(batch, target_layers, targets, model, method):
    # === Compute Grad-CAM ===
        # Set aug_smooth=True and eigen_smooth=True for smoother maps (optional)
    with method(model=model, target_layers=target_layers) as cam:
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
def compare_visualizations(orig_image_np, grayscale_cam_orig, pred_orig, adv_image_np, grayscale_cam_adv, pred_adv, method):
    orig_image_np = denormalize_np(orig_image_np)                       # convert to [0, 1] range for visualization
    adv_image_np = denormalize_np(adv_image_np)

    orig_visualization = show_cam_on_image(orig_image_np, grayscale_cam_orig, use_rgb=True)
    adv_visualization = show_cam_on_image(adv_image_np, grayscale_cam_adv, use_rgb=True)
    method_name = str(method).split(".")[-1].split("\'")[0]

    # === Display using Matplotlib ===
    fig_comp = plt.figure(figsize=(12, 8)) # Adjusted size a bit

    plt.subplot(2, 2, 1)
    plt.imshow(orig_image_np)
    plt.title("Original Image\n")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(orig_visualization)
    plt.title(f"{method_name}\nTarget/Pred: {pred_orig}")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(adv_image_np)
    plt.title("Adversarial Image\n")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(adv_visualization)
    plt.title(f"{method_name}\nTarget/Pred: {pred_adv}")
    plt.axis("off")

    plt.suptitle(f"Comparison for {method_name}", fontsize=14) # Add suptitle for context
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.5, w_pad=0.5) # Adjusted padding
    # print(f"{method_name} comparison visualization complete.")
    return fig_comp

''' function to calculate the difference in explanation for original and adversarial image
arguments: grayscale_cam_orig: pixel importances for original image, 
grayscale_cam_adv: pixel importances for adversarial input
returns: mean of pixelwise differences'''
def mean_pixel_difference(grayscale_cam_orig, grayscale_cam_adv):
    return np.mean(np.abs(grayscale_cam_orig - grayscale_cam_adv))

''' function to count the number of pixels that are significantly different for both explanations
arguments: grayscale_cam_orig: pixel importances for original image, 
grayscale_cam_adv: pixel importances for adversarial input
returns: number of pixels, where the importance is significantly different (difference greater than 0.1)'''
def pixel_differrences_count(grayscale_cam_orig, grayscale_cam_adv):
    differences = np.abs(grayscale_cam_orig - grayscale_cam_adv)
    return np.count_nonzero(differences > 0.2)

''' function to measure how the number of important pixels changes for the original and adversarial image based on a threshold'''
def pixel_activation_ratio(grayscale_cam_orig, grayscale_cam_adv, threshold=0):
    pixels_activated_orig = np.count_nonzero(grayscale_cam_orig > threshold)
    pixels_activated_adv = np.count_nonzero(grayscale_cam_adv > threshold)
    return pixels_activated_adv/pixels_activated_orig

''' function to count the number of regions (connected components) that are considered highly relevant 
(pixel importance higher than 0.9)'''
def count_attention_regions(saliency_map: np.ndarray, threshold: float = 0.9) -> int:
    # Thresholding: keep only high-importance pixels
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

''' function to compute the Jenson-Shannon Divergence (symmetric distance for probability distributions)'''
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

''' function to compute the intersection over union for the saliency maps of original and adversarial image'''
def compute_iou(saliency_map_orig, saliency_map_adv, threshold_a=0.8, threshold_b=0.8):
    A_mask = saliency_map_orig > np.quantile(saliency_map_orig, threshold_a)
    B_mask = saliency_map_adv > np.quantile(saliency_map_adv, threshold_b)

    # Calculate IoU
    intersection = np.logical_and(A_mask, B_mask).sum()
    union = np.logical_or(A_mask, B_mask).sum()

    return intersection / union if union > 0 else 0.0


''' function to calculate metrics to compare the original saliency map and the one for the adversarial.
arguments: grayscale_cam_orig: pixel importances of original image,
           grayscale_cam_adv: pixel importances of adversarial
returns: mean of absolute pixel differences, percentage of pixels different according to threshold (0.2), cosine similarity, activation ratio'''
def calculate_metrics(grayscale_cam_orig, grayscale_cam_adv):
    mean_pixel_diff = mean_pixel_difference(grayscale_cam_orig, grayscale_cam_adv)
    percent_different_pixels = pixel_differrences_count(grayscale_cam_orig, grayscale_cam_adv) / (image.shape[1] * image.shape[2]) * 100
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

''' function to check if the model makes the correct classification and if it does creating adversarial image
arguments: preprocessed image (Tensor of size [3, 224, 224]), label (Tensor of size [1] containing class_id),
attack: Class from foolbox.attacks, epsilons: numpy array of epsilons to try
returns: adv_image: adversarial image (Tensor of size [1, 3, 224, 224]) (None if no adversarial was produced or classification is not correct)'''
def check_classification_and_create_untargeted_adversarials(image, label, attack, epsilons):
    pred, score = make_prediction(model, image)
    # check if prediction is correct
    if pred != weights.meta["categories"][label]:
        print("Classification not correct")
        return None
    
    # TODO: Targeted Attack if possible (not supported by all attacks)
    #target_labels = torch.tensor(400).unsqueeze(0)
    #criterion = fb.criteria.TargetedMisclassification(target_labels)

    raw, clipped, is_adv = attack(fmodel, image.unsqueeze(0), label.unsqueeze(0), epsilons=epsilons)
    if epsilons is not None:
    # Multi-epsilon attack
        indices = torch.nonzero(is_adv, as_tuple=False)
        if indices.numel() > 0:
            first_true_index = indices[0][0].item()                         # choose smallest epsilon creating an adversarial image
            adv_image = clipped[first_true_index]
            used_epsilon = epsilons[first_true_index]
            print(f"Used epsilon: {used_epsilon}")
        else:
            print("No adversarial image was produced.")
            return None
    else:
        # Single output attack (e.g., Pointwise)
        if is_adv.item():
            adv_image = clipped
            print("Adversarial image was produced (no epsilon used).")
        else:
            print("No adversarial image was produced.")
            return None
    return adv_image

''' function to run the explanation on the original image and compare the explanation for the adversarial with the original one
saves generated plots (comparison of heatmaps) and stores metrics of comparison in csv file
arguments: preprocessed image (Tensor of size [3, 224, 224]), label (Tensor of size [1] containing class_id),
xAImethod: class from pytorch-gradcam, adv_image: Tensor of size [1, 3, 224, 224]'''
def create_and_compare_explanations(image, label, xAImethod, adv_image, csv_file):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    pred, score = make_prediction(model, image)

    targets, target_layers = get_targets(model, pred)

    pred_adv, score_adv = make_prediction(model, adv_image[0])
    targets_adv, target_layers_adv = get_targets(model, pred_adv)
    batch = torch.stack([image, adv_image[0]], dim=0)
    grayscale_cams = calculate_multiple_explanations(batch, target_layers + target_layers_adv, targets + targets_adv, model, xAImethod)
    grayscale_cam_orig = grayscale_cams[0, :]
    grayscale_cam_adv = grayscale_cams[1, :]
    fig_to_save = compare_visualizations(image_np, grayscale_cam_orig, pred, adv_image[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam_adv, pred_adv, xAImethod)

    ### saving figure
    method_name = str(xAImethod).split(".")[-1].split("\'")[0]
    attack_name = str(attack).split("(")[0]
    class_name = weights.meta["categories"][label]
    fig_to_save.savefig("results/images/" + method_name + "_" + attack_name + "_" + class_name + str(idx) + ".png")
    plt.close(fig_to_save)
    images_similarity = torch.mean((image - adv_image)**2).item()
    print(f"Difference of images: {images_similarity}")

    ### saving metrics
    mean_pixel_diff, percent_different_pixels, cosine_sim, activation_ratio, highly_relevant_ratio, js_div, num_regions_orig, num_regions_adv, intersection_over_union = calculate_metrics(grayscale_cam_orig, grayscale_cam_adv)


    # Write header if the file does not exist
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow([
                "idx", "method_name", "attack_name", "class_name", "target_pred", "images_mean_difference", "score_original", "score_adversarial",
                "mean_pixel_diff", "percent_different_pixels", "num_regions_orig", "num_regions_adv",
                "cosine_sim", "activation_ratio", "js_div", "highly_relevant_pixels_ratio", "intersection_over_union"
            ])
        writer.writerow([
            idx, method_name, attack_name, class_name, pred_adv, images_similarity, score, score_adv,
            mean_pixel_diff, percent_different_pixels, num_regions_orig, num_regions_adv, cosine_sim, 
            activation_ratio, js_div, highly_relevant_ratio, intersection_over_union
        ])
    del grayscale_cam_adv, grayscale_cam_orig, grayscale_cams, batch


''' function to load images from imagenet subset and preprocess them
returns: tensor of images and tensor of labels (class_ids)'''
def load_and_transform_images(n=10):
    images = []
    labels = []
    dataset = load_dataset("ioxil/imagenetsubset")
    test_data = dataset["train"]

    for sample in test_data:
        image = sample["image"].convert("RGB")
        class_id = sample["label"]
        tensor_image = preprocess(image)
        images.append(tensor_image)
        labels.append(torch.tensor(class_id))

    return torch.stack(images), torch.tensor(labels)


if __name__ == '__main__':

    # Check if argument is provided
    if len(sys.argv) < 2:
        print("Error: Missing argument. Usage: python pipeline.py <index>")
        sys.exit(1)

    # Get the argument (e.g., 0, 1, 2, or 3)
    attack_group_index = int(sys.argv[1])

    print(f"Running pipeline with index: {attack_group_index}")

    # Setup Folders
    results_folder = 'results'
    images_folder = os.path.join(results_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    print(f"Results will be saved in: {results_folder}")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Using pretrained weights and model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).eval()
    model = model.to(device)

    preprocess = weights.transforms()
    bounds = (-2.4, 2.8)
    # preprocessing None as images from fb.utils.samples seem to be already preprocessed
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)

    images, labels = load_and_transform_images()
    #images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)

    images = images.to(device)
    labels = labels.to(device)
    
    print(fb.utils.accuracy(fmodel, images, labels))

    print(labels.unique(return_counts=True))
                   
    attack_to_epsilon = [{
    fb_att.LinfFastGradientAttack(): np.linspace(0, 0.5, num=20),
    fb_att.LinfProjectedGradientDescentAttack(): np.linspace(0, 0.05, num=5),
    fb_att.L2FastGradientAttack(): np.linspace(1, 80, num=60),
    fb_att.L2ProjectedGradientDescentAttack(): np.linspace(0, 5, num=20),
    fb_att.LInfFMNAttack(): np.linspace(0, 0.5, num=20),
    fb_att.L2FMNAttack(): np.linspace(0, 2, num=10),
    fb_att.L1FMNAttack(): np.linspace(2, 100, num=20),
    fb_att.LinfinityBrendelBethgeAttack(steps=200): np.linspace(0, 0.5, num=10),
    },                                                         

    {fb_att.L2BrendelBethgeAttack(): np.linspace(0, 1, num=15),
    fb_att.L1BrendelBethgeAttack(steps=200): np.linspace(2, 100, num=10),
    fb_att.LinearSearchBlendedUniformNoiseAttack(distance=LpDistance(100)): np.linspace(0, 20, num=20),                       # (very) perturbed images
    fb_att.SaltAndPepperNoiseAttack(): np.linspace(1, 200, num=20),
    fb_att.LinfDeepFoolAttack(): np.linspace(0, 0.5, num=10),
    fb_att.L2DeepFoolAttack(): np.linspace(0, 1, num=20),
    fb_att.GaussianBlurAttack(distance=LpDistance(2)): np.linspace(1, 150, num=20)
    },

    { 
    fb_att.LinfRepeatedAdditiveUniformNoiseAttack(): np.linspace(0.1, 1, num=20),   
    fb_att.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(): np.linspace(1, 150, num=20),
    fb_att.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(): np.linspace(1, 150, num=20),
    fb_att.L2RepeatedAdditiveUniformNoiseAttack(): np.linspace(1, 150, num=20),
    fb_att.L2RepeatedAdditiveGaussianNoiseAttack(): np.linspace(20, 150, num=30),
    fb_att.LinfAdditiveUniformNoiseAttack(): np.linspace(0, 2, num=40),
    fb_att.LinfBasicIterativeAttack(): np.linspace(0, 0.05, num=10),
    fb_att.BoundaryAttack(steps=10000): np.linspace(1, 50, num=10)                              # very slow
 },

    { fb_att.L2ClippingAwareAdditiveUniformNoiseAttack(): np.linspace(1, 250, num=20),
    fb_att.L2ClippingAwareAdditiveGaussianNoiseAttack(): np.linspace(1, 250, num=20),
    fb_att.L2AdditiveUniformNoiseAttack(): np.linspace(1, 250, num=20),
    fb_att.L2AdditiveGaussianNoiseAttack(): np.linspace(1, 250, num=20),
    fb_att.VirtualAdversarialAttack(steps=50): np.linspace(10, 100, num=50),
    fb_att.DDNAttack(): np.linspace(0, 1, num=20),
    fb_att.L2BasicIterativeAttack(): np.linspace(0, 4, num=30),
    fb_att.EADAttack(steps=5000): np.linspace(1, 100, num=10),                                                               # very slow
    fb_att.NewtonFoolAttack(): np.linspace(0, 10, num=10),
    fb_att.L2CarliniWagnerAttack(steps=1000): np.linspace(0, 10, num=10)
    } ]

    all_attacks = attack_to_epsilon[attack_group_index]

    # Define the path to the CSV file
    csv_file = "results/cam_comparison_metrics" + str(attack_group_index) + ".csv"


    explanation_methods = [GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, 
                            EigenCAM, EigenGradCAM, LayerCAM, FullGrad, KPCA_CAM, AblationCAM, ScoreCAM]


    for attack, epsilons in all_attacks.items():
        num_adv_failed = 0
        print(f"Doing attack: {attack}")
        ## create adversarial images then use explanation methods
        for idx, image in enumerate(images):              
            adv_image = check_classification_and_create_untargeted_adversarials(images[idx], labels[idx], attack, epsilons)
            if adv_image is None:
                num_adv_failed += 1
                continue

            for xAImethod in explanation_methods:
                print(f"Trying explanation method: {xAImethod}")
                create_and_compare_explanations(images[idx], labels[idx], xAImethod, adv_image, csv_file)

            del adv_image
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print(f"  GPU Mem (After Image {idx} Proc.): Alloc: {torch.cuda.memory_allocated(device)/(1024**2):.1f}MB, MaxAlloc: {torch.cuda.max_memory_allocated(device)/(1024**2):.1f}MB, Cached: {torch.cuda.memory_reserved(device)/(1024**2):.1f}MB")
        print(f"Failed to create adversarial in {num_adv_failed} cases. For attack: {attack}. \n")
            




## attacks that do not work good: 

# fb_att.LinearSearchContrastReductionAttack(distance=LpDistance(2)): np.linspace(20, 200, num=15),                       # images are almost only gray 
#     fb_att.BinarySearchContrastReductionAttack(distance=LpDistance(2)): np.linspace(20, 200, num=15),                        # images are almost only gray   
#     fb_att.InversionAttack(distance=LpDistance(2)): np.linspace(20, 200, num=20),                                            # images are almost only gray
    # fb_att.L2ContrastReductionAttack(): np.linspace(50, 300, num=20),                                                    # images are almost only gray 
