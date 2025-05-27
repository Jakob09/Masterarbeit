from torchvision.models import resnet50, ResNet50_Weights
import os
import gc
import csv
import torch
import numpy as np
import torchvision.io as io
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM
from pytorch_grad_cam import ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization, ShapleyCAM, FinerCAM, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import foolbox as fb
import scipy.stats # type: ignore
import scipy.ndimage # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import foolbox.attacks as fb_att
from foolbox.distances import LpDistance
from scipy.special import kl_div
from datasets import load_dataset # type: ignore

# --- Denormalization functions (unchanged) ---
def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    mean_t = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std_t + mean_t

def denormalize_np(image_np):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    image_np = image_np.astype(np.float32)
    mean_np = np.array(mean).reshape(1, 1, 3)
    std_np = np.array(std).reshape(1, 1, 3)
    return np.clip((image_np * std_np + mean_np), 0, 1)

# --- Target and Prediction functions ---
def get_targets(model, prediction_category_name): # Pass category name directly
    weights = ResNet50_Weights.DEFAULT # Assuming ResNet50_Weights.DEFAULT is globally accessible or passed
    class_id = weights.meta["categories"].index(prediction_category_name)
    target_layers = [model.layer4[-1]] # Assuming model is accessible
    targets = [ClassifierOutputTarget(class_id)]
    return targets, target_layers

def make_prediction(model_instance, img_tensor): # model_instance to avoid conflict with global 'model'
    weights = ResNet50_Weights.DEFAULT # Assuming ResNet50_Weights.DEFAULT is accessible
    device = next(model_instance.parameters()).device
    img_tensor_device = img_tensor.to(device)

    with torch.no_grad(): # Add torch.no_grad()
        batch = img_tensor_device.unsqueeze(0)
        prediction_softmax = model_instance(batch).squeeze(0).softmax(0)
        class_id = prediction_softmax.argmax().item()
        score = prediction_softmax[class_id].item()
    category_name = weights.meta["categories"][class_id]
    # print(f"Prediction: {category_name}: {100 * score:.1f}%") # Keep print for debugging if needed
    return category_name, score

# --- CAM Calculation and Visualization (largely unchanged, ensure inputs are correct) ---
def calculate_explanation(img_tensor, target_layers, targets, model_instance, method_class):
    batch = img_tensor.unsqueeze(0)
    device = next(model_instance.parameters()).device
    batch = batch.to(device)
    with method_class(model=model_instance, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=batch,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

def calculate_multiple_explanations(batch_tensor, target_layers_list, targets_list, model_instance, method_class):
    device = next(model_instance.parameters()).device
    batch_tensor = batch_tensor.to(device)
    # Ensure target_layers_list is appropriate (usually one set for the model)
    # targets_list should be a list of target objects, one per image in the batch
    with method_class(model=model_instance, target_layers=target_layers_list[0] if isinstance(target_layers_list[0], list) else target_layers_list) as cam: # Use first set of target layers if multiple provided, or ensure it's single
        grayscale_cam = cam(input_tensor=batch_tensor,
                            targets=targets_list)
        return grayscale_cam

def visualize_explanation(image_np_normalized, grayscale_cam, prediction_str, method_class): # image_np should be normalized HWC float32
    method_name = str(method_class).split(".")[-1].split("\'")[0]
    # denormalize_np will be called here
    visualization = show_cam_on_image(denormalize_np(image_np_normalized), grayscale_cam, use_rgb=True)
    
    fig_vis, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(denormalize_np(image_np_normalized)) # Show denormalized original for direct comparison
    axes[0].set_title("Original Image\n")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title(f"{method_name}\nTarget/Pred: {prediction_str}")
    axes[1].axis('off')

    fig_vis.suptitle(f"{method_name} for ResNet50")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show() # Usually commented out in scripts
    # print(f"{method_name} visualization complete.")
    return fig_vis # Return figure object to be closed by caller

def compare_visualizations(orig_image_np_norm, grayscale_cam_orig, pred_orig_str,
                           adv_image_np_norm, grayscale_cam_adv, pred_adv_str, method_class):
    # Inputs are normalized numpy arrays [H,W,C]
    orig_image_denorm = denormalize_np(orig_image_np_norm)
    adv_image_denorm = denormalize_np(adv_image_np_norm)

    orig_visualization = show_cam_on_image(orig_image_denorm, grayscale_cam_orig, use_rgb=True)
    adv_visualization = show_cam_on_image(adv_image_denorm, grayscale_cam_adv, use_rgb=True)
    method_name = str(method_class).split(".")[-1].split("\'")[0]

    fig_comp = plt.figure(figsize=(12, 8)) # Adjusted size a bit

    plt.subplot(2, 2, 1)
    plt.imshow(orig_image_denorm)
    plt.title("Original Image\n")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(orig_visualization)
    plt.title(f"{method_name}\nTarget/Pred: {pred_orig_str}")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(adv_image_denorm)
    plt.title("Adversarial Image\n")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(adv_visualization)
    plt.title(f"{method_name}\nTarget/Pred: {pred_adv_str}")
    plt.axis("off")

    plt.suptitle(f"Comparison for {method_name}", fontsize=14) # Add suptitle for context
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.5, w_pad=0.5) # Adjusted padding
    # print(f"{method_name} comparison visualization complete.")
    return fig_comp # Return figure object

# --- Metrics (pass total_pixels_in_cam) ---
def mean_pixel_difference(grayscale_cam_orig, grayscale_cam_adv):
    return np.mean(np.abs(grayscale_cam_orig - grayscale_cam_adv))

def pixel_differrences_count(grayscale_cam_orig, grayscale_cam_adv):
    differences = np.abs(grayscale_cam_orig - grayscale_cam_adv)
    return np.count_nonzero(differences > 0.2)

def pixel_activation_ratio(grayscale_cam_orig, grayscale_cam_adv, threshold=0):
    pixels_activated_orig = np.count_nonzero(grayscale_cam_orig > threshold)
    pixels_activated_adv = np.count_nonzero(grayscale_cam_adv > threshold)
    if pixels_activated_orig == 0: return 0 # Avoid division by zero
    return pixels_activated_adv / pixels_activated_orig

def count_attention_regions(saliency_map: np.ndarray, threshold: float = 0.9) -> int:
    binary_map = (saliency_map >= threshold).astype(np.uint8)
    _, num_regions = scipy.ndimage.label(binary_map, structure=np.ones((3, 3)))
    return num_regions

def normalize_saliency_map(saliency_map: np.ndarray):
    saliency_map_norm = saliency_map + 1e-10 # ensure non-zero for KL
    return saliency_map_norm / np.sum(saliency_map_norm)

def compute_js_divergence(saliency1: np.ndarray, saliency2: np.ndarray):
    saliency1_prob = normalize_saliency_map(saliency1)
    saliency2_prob = normalize_saliency_map(saliency2)
    m = 0.5 * (saliency1_prob + saliency2_prob)
    js_div = 0.5 * (np.sum(kl_div(saliency1_prob, m)) + np.sum(kl_div(saliency2_prob, m)))
    return js_div

def compute_iou(saliency_map_orig, saliency_map_adv, threshold_quantile=0.8):
    # Using quantile for thresholding makes it adaptive to the map's value distribution
    threshold_a_val = np.quantile(saliency_map_orig, threshold_quantile)
    threshold_b_val = np.quantile(saliency_map_adv, threshold_quantile)
    
    A_mask = saliency_map_orig > threshold_a_val
    B_mask = saliency_map_adv > threshold_b_val

    intersection = np.logical_and(A_mask, B_mask).sum()
    union = np.logical_or(A_mask, B_mask).sum()
    return intersection / union if union > 0 else 0.0

def calculate_metrics(grayscale_cam_orig, grayscale_cam_adv, total_pixels_in_cam): # Pass total_pixels_in_cam
    mean_pix_diff = mean_pixel_difference(grayscale_cam_orig, grayscale_cam_adv)
    
    if total_pixels_in_cam == 0: # Safeguard
        percent_diff_pixels = 0.0
    else:
        percent_diff_pixels = pixel_differrences_count(grayscale_cam_orig, grayscale_cam_adv) / total_pixels_in_cam * 100
    
    cos_sim = cosine_similarity(grayscale_cam_orig.flatten().reshape(1, -1), grayscale_cam_adv.flatten().reshape(1, -1))[0, 0]
    act_ratio = pixel_activation_ratio(grayscale_cam_orig, grayscale_cam_adv)
    js_d = compute_js_divergence(grayscale_cam_orig, grayscale_cam_adv)
    highly_rel_ratio = pixel_activation_ratio(grayscale_cam_orig, grayscale_cam_adv, threshold=0.7)
    iou = compute_iou(grayscale_cam_orig, grayscale_cam_adv)
    num_regions_o = count_attention_regions(grayscale_cam_orig, threshold=0.8)
    num_regions_a = count_attention_regions(grayscale_cam_adv, threshold=0.8)

    # print(f"  Metrics: Mean Pix Diff: {mean_pix_diff:.4f}, % Sig Diff Pix: {percent_diff_pixels:.2f}%, Cos Sim: {cos_sim:.3f}")
    # print(f"  Act Ratio: {act_ratio:.3f}, Highly Rel Ratio: {highly_rel_ratio:.3f}, JS Div: {js_d:.4f}, IoU: {iou:.3f}")
    # print(f"  Regions Orig: {num_regions_o}, Regions Adv: {num_regions_a}")
    return mean_pix_diff, percent_diff_pixels, cos_sim, act_ratio, highly_rel_ratio, js_d, num_regions_o, num_regions_a, iou

# --- Adversarial Attack Generation (Crucial for memory) ---
def check_classification_and_create_untargeted_adversarials(fmodel_instance, model_for_pred, # Pass fmodel and model
                                                           image_tensor_orig, label_tensor_orig, # original tensors
                                                           attack_obj, epsilons_options, resnet_weights): # attack instance and its epsilons
    
    device = next(fmodel_instance.model.parameters()).device # Get device from fmodel
    current_image = image_tensor_orig.to(device)
    current_label = label_tensor_orig.to(device) # Ensure label is also a tensor on device

    pred_category_name, _ = make_prediction(model_for_pred, current_image)
    true_category_name = resnet_weights.meta["categories"][current_label.item()]

    if pred_category_name != true_category_name:
        print(f"  Initial classification incorrect. Pred: {pred_category_name}, True: {true_category_name}. Skipping attack.")
        return None

    adv_image_final = None
    used_epsilon_val = "N/A" # For attacks without explicit epsilons list

    # Foolbox expects batch of images and labels
    img_batch = current_image.unsqueeze(0)
    label_batch = current_label.unsqueeze(0) # This must be a tensor for foolbox

    try:
        raw, clipped_candidates, is_adv = attack_obj(fmodel_instance, img_batch, label_batch, epsilons=epsilons_options)
        
        if epsilons_options is not None:
            # is_adv can be a boolean tensor [num_epsilons]
            successful_adv_indices = torch.nonzero(is_adv, as_tuple=False)
            if successful_adv_indices.numel() > 0:
                first_successful_idx = successful_adv_indices[0][0].item()
                # CLONE the tensor to ensure it's a new independent copy
                adv_image_final = clipped_candidates[first_successful_idx].clone()
                used_epsilon_val = epsilons_options[first_successful_idx]
                # print(f"  Adversarial produced with epsilon: {used_epsilon_val}")
            else:
                # print("  No adversarial image produced with the given epsilons.")
                pass
        else: # For attacks that don't iterate over an epsilon list (e.g., BoundaryAttack)
            if is_adv.item(): # is_adv should be a scalar boolean tensor or single element tensor
                adv_image_final = clipped_candidates.clone() # CLONE
                # print("  Adversarial image produced (no epsilon list used).")
            else:
                # print("  No adversarial image produced (no epsilon list used).")
                pass
        
        # Explicitly delete large intermediate tensors from the attack
        del raw, clipped_candidates, is_adv

    except Exception as e:
        print(f"  ERROR during adversarial attack {attack_obj.__class__.__name__}: {e}")
        # Ensure cleanup even on error
        if 'raw' in locals(): del raw
        if 'clipped_candidates' in locals(): del clipped_candidates
        if 'is_adv' in locals(): del is_adv
        return None
    
    if adv_image_final is not None:
        # print(f"  Successfully generated adversarial. Epsilon: {used_epsilon_val}")
        return adv_image_final # This is a [C, H, W] tensor on GPU
    else:
        # print("  Failed to generate adversarial.")
        return None


# --- Main Comparison Logic (Crucial for memory) ---
def create_and_compare_explanations(current_image_idx, # For logging and filenames
                                    model_instance, resnet_weights, # Pass model and weights
                                    original_image_gpu, original_label_gpu, # Original data on GPU
                                    adv_image_gpu, # Adversarial image [C,H,W] on GPU
                                    xAImethod_class, attack_name_str, # Method class and attack name string
                                    csv_writer_obj): # Pass CSV writer object

    # Prepare images for CAM and visualization
    # Images are already [C,H,W] on GPU
    original_image_np_norm = original_image_gpu.permute(1, 2, 0).cpu().numpy() # For viz
    adv_image_np_norm = adv_image_gpu.permute(1, 2, 0).cpu().numpy() # For viz

    # Predictions
    pred_orig_str, score_orig = make_prediction(model_instance, original_image_gpu)
    pred_adv_str, score_adv = make_prediction(model_instance, adv_image_gpu)

    # Get targets for CAM
    targets_orig, target_layers_orig = get_targets(model_instance, pred_orig_str)
    targets_adv, target_layers_adv = get_targets(model_instance, pred_adv_str) # Targets might change if prediction changes

    # Batch for CAM. target_layers_orig should be fine as it's model-specific layer
    batch_for_cam = torch.stack([original_image_gpu, adv_image_gpu], dim=0)
    
    # Assuming target_layers are the same for the model (e.g. model.layer4[-1])
    # And targets_orig/targets_adv are lists like [ClassifierOutputTarget(class_id)]
    cam_targets_list = targets_orig + targets_adv 

    try:
        grayscale_cams_batch = calculate_multiple_explanations(batch_for_cam, target_layers_orig, cam_targets_list, model_instance, xAImethod_class)
        grayscale_cam_orig = grayscale_cams_batch[0, :].cpu().numpy() # Move to CPU and NumPy for metrics/viz
        grayscale_cam_adv = grayscale_cams_batch[1, :].cpu().numpy()
    except Exception as e:
        print(f"  ERROR during CAM ({xAImethod_class.__name__}) calculation: {e}")
        del batch_for_cam # Clean up batch if CAM fails
        return # Skip this explanation method if CAM fails

    # Visualization
    method_name_str = str(xAImethod_class).split(".")[-1].split("\'")[0]
    fig_to_save = compare_visualizations(original_image_np_norm, grayscale_cam_orig, pred_orig_str,
                                         adv_image_np_norm, grayscale_cam_adv, pred_adv_str, xAImethod_class)
    
    # Saving figure
    true_class_name_str = resnet_weights.meta["categories"][original_label_gpu.item()]
    # Ensure results/images directory exists (should be done once at start)
    save_path = f"results/images/{method_name_str}_{attack_name_str}_{true_class_name_str}_{current_image_idx}.png"
    fig_to_save.savefig(save_path)
    plt.close(fig_to_save) # VERY IMPORTANT: Close the figure

    # Image similarity (MSE)
    img_mse_diff = torch.mean((original_image_gpu - adv_image_gpu)**2).item()

    # Calculate metrics
    cam_h, cam_w = grayscale_cam_orig.shape
    total_cam_pixels = cam_h * cam_w
    metrics_tuple = calculate_metrics(grayscale_cam_orig, grayscale_cam_adv, total_cam_pixels)
    
    # Writing to CSV
    csv_writer_obj.writerow([
        current_image_idx, method_name_str, attack_name_str, true_class_name_str, pred_adv_str, img_mse_diff,
        score_orig, score_adv] + list(metrics_tuple)
    )

    # Clean up tensors created in this function scope
    del grayscale_cams_batch, grayscale_cam_orig, grayscale_cam_adv, batch_for_cam
    del original_image_np_norm, adv_image_np_norm 
    # original_image_gpu, adv_image_gpu, original_label_gpu are managed by the caller loop.
    # gc.collect() and torch.cuda.empty_cache() will be called in the main loop.


# --- Data Loading (Unchanged, but ensure it's efficient if dataset is huge) ---
def load_and_transform_images(n_samples=10, image_size=224): # Added image_size
    images_cpu = []
    labels_cpu = []
    dataset = load_dataset("ioxil/imagenetsubset", split=f'train[:{n_samples}]') # Load only n_samples

    # Get the preprocessing transforms from ResNet50_Weights
    # This needs to be defined globally or passed if used here
    # For now, assuming global `weights_global` for ResNet50_Weights.DEFAULT
    preprocess_transform = weights_global.transforms()

    for sample in dataset:
        image = sample["image"].convert("RGB") # Ensure RGB
        # Apply the standard ResNet50 preprocessing
        tensor_image = preprocess_transform(image)
        images_cpu.append(tensor_image)
        labels_cpu.append(torch.tensor(sample["label"])) # Class ID as tensor

    if not images_cpu: # Handle empty dataset case
        return torch.empty(0), torch.empty(0)

    return torch.stack(images_cpu), torch.stack(labels_cpu) # Stack creates [N,C,H,W] and [N]


# --- Main Script ---
if __name__ == '__main__':
    # Setup Folders
    results_folder = 'results'
    images_folder = os.path.join(results_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    print(f"Results will be saved in: {results_folder}")

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Global Model and Weights (referenced by helper functions)
    weights_global = ResNet50_Weights.DEFAULT
    model_global = resnet50(weights=weights_global).eval().to(device)
    # Preprocessing for Foolbox (bounds of normalized images)
    # Min: (0 - max_mean) / min_std approx (0-0.485)/0.229 = -2.117
    # Max: (1 - min_mean) / min_std approx (1-0.406)/0.225 = 2.64
    fb_bounds = (-2.2, 2.7) 
    fmodel_global = fb.PyTorchModel(model_global, bounds=fb_bounds, device=device) # Pass device

    # Load Data (adjust n_samples as needed)
    # For testing memory, start with a very small number of images.
    num_images_to_load = 10 # Example: use 2-3 for quick memory test runs
    all_images_cpu, all_labels_cpu = load_and_transform_images(n_samples=num_images_to_load)
    
    if all_images_cpu.numel() == 0:
        print("No images loaded. Exiting.")
        exit()

    all_images_gpu = all_images_cpu.to(device)
    all_labels_gpu = all_labels_cpu.to(device)
    print(f"Loaded {all_images_gpu.shape[0]} images to {device}.")

    # Initial accuracy check (optional)
    # print(f"Initial accuracy: {fb.utils.accuracy(fmodel_global, all_images_gpu, all_labels_gpu)}")

    # Attacks and Epsilon Definitions (shortened for brevity, use your full lists)
    epsilons_map = { # Using a dictionary for easier mapping
        fb_att.LinfFastGradientAttack: np.linspace(0, 0.05, num=5), # Reduced num for faster test
        fb_att.L2FastGradientAttack: np.linspace(0, 1.0, num=5),
        fb_att.LinfProjectedGradientDescentAttack: np.linspace(0, 0.05, num=5),
        # ... add all your attacks and their corresponding epsilon lists
        # Example: Using a default if not specified, or None for attacks that don't need an epsilon list
    }
    default_epsilons = np.linspace(0, 0.1, num=5) # A generic default

    attacks_to_run = [
        fb_att.LinfFastGradientAttack(),
        fb_att.L2FastGradientAttack(),
        fb_att.LinfProjectedGradientDescentAttack(steps=10),
        fb_att.BoundaryAttack(steps=100), # Note: BoundaryAttack is slow and uses epsilons=None
        # ... add more attack instances
    ]

    explanation_methods_to_run = [GradCAM, HiResCAM, GradCAMPlusPlus, LayerCAM, EigenCAM] # Select a few for testing

    # CSV Setup
    csv_file_path = os.path.join(results_folder, "cam_comparison_metrics.csv")
    csv_file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not csv_file_exists:
            csv_writer.writerow([
                "image_idx", "method_name", "attack_name", "true_class", "adv_class", "img_mse_diff",
                "score_orig", "score_adv", "mean_pixel_diff", "percent_diff_pixels", 
                "cosine_sim", "activation_ratio", "highly_rel_ratio", "js_div", 
                "num_regions_orig", "num_regions_adv", "iou"
            ])

        num_adv_failed = 0
        total_main_loops = 0

        for attack_obj_instance in attacks_to_run:
            attack_name = attack_obj_instance.__class__.__name__
            print(f"\n>>> Starting Attack: {attack_name}")
            
            # Get epsilons for this attack, BoundaryAttack typically doesn't use an epsilon list for generation
            current_attack_epsilons = None
            if not isinstance(attack_obj_instance, fb_att.BoundaryAttack): # Example specific handling
                 current_attack_epsilons = epsilons_map.get(type(attack_obj_instance), default_epsilons)


            for img_idx in range(all_images_gpu.shape[0]):
                total_main_loops += 1
                original_img_gpu_single = all_images_gpu[img_idx] # [C,H,W]
                original_lbl_gpu_single = all_labels_gpu[img_idx]   # scalar tensor

                print(f"\n  Processing Image Idx: {img_idx} | Attack: {attack_name} | Loop {total_main_loops}")
                
                # Print initial GPU memory for this image processing cycle
                if device.type == 'cuda':
                    print(f"  GPU Mem (Before Adv): Alloc: {torch.cuda.memory_allocated(device)/(1024**2):.1f}MB, MaxAlloc: {torch.cuda.max_memory_allocated(device)/(1024**2):.1f}MB, Cached: {torch.cuda.memory_reserved(device)/(1024**2):.1f}MB")


                adv_img_gpu_single = check_classification_and_create_untargeted_adversarials(
                    fmodel_global, model_global, original_img_gpu_single, original_lbl_gpu_single,
                    attack_obj_instance, current_attack_epsilons, weights_global
                )

                if adv_img_gpu_single is None:
                    num_adv_failed += 1
                    print(f"  Failed to create adversarial for image {img_idx} with {attack_name}. Skipping.")
                    # Clean up any potential lingering references even on failure
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()
                    continue
                
                # adv_img_gpu_single is [C,H,W] and on GPU

                for xai_method_cls in explanation_methods_to_run:
                    xai_method_name = xai_method_cls.__name__
                    print(f"    Running Explanation: {xai_method_name}")
                    try:
                        create_and_compare_explanations(
                            img_idx, model_global, weights_global,
                            original_img_gpu_single, original_lbl_gpu_single,
                            adv_img_gpu_single, # Pass the [C,H,W] tensor
                            xai_method_cls, attack_name, csv_writer
                        )
                    except Exception as e_exp:
                        print(f"    ERROR during explanation {xai_method_name} for image {img_idx}: {e_exp}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # Crucial cleanup after each explanation method for an image
                        gc.collect()
                        if device.type == 'cuda': torch.cuda.empty_cache()
                
                # Clean up adversarial image tensor after all explanations for it are done
                del adv_img_gpu_single
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    print(f"  GPU Mem (After Image {img_idx} Proc.): Alloc: {torch.cuda.memory_allocated(device)/(1024**2):.1f}MB, MaxAlloc: {torch.cuda.max_memory_allocated(device)/(1024**2):.1f}MB, Cached: {torch.cuda.memory_reserved(device)/(1024**2):.1f}MB")


    print(f"\n--- Script Finished ---")
    print(f"Total adversarial generation attempts failed: {num_adv_failed}")
    if device.type == 'cuda':
        print(f"Final Max GPU Memory Allocated: {torch.cuda.max_memory_allocated(device)/(1024**2):.1f}MB")
        torch.cuda.reset_peak_memory_stats(device) # Reset for next potential run if in interactive session