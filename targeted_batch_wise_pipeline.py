from batch_wise_pipeline import *


''' function to check if the model makes the correct classification and if it does creating adversarial image
arguments: batch of preprocessed images (Tensor of size [N, 3, 224, 224]), label (Tensor of size [N] containing the class_ids),
attack: Class from foolbox.attacks, epsilons: numpy array of epsilons to try for that attack
returns: selected_advs: adversarial images (Tensor of size [N, 3, 224, 224]) (None if no adversarial was produced)'''
def create_targeted_adversarials(image_batch, label_batch, attack, epsilons):

    target_labels = (label_batch + 300) % 1000
    criterion = fb.criteria.TargetedMisclassification(target_labels)

    raw, clipped, is_adv = attack(fmodel, image_batch, criterion, epsilons=epsilons)                      # clipped: list of tensors of size [N, 3, 224, 224] -> index of list corresponds to epsilon
                                                                                                            # different images might have different best epsilons
    selected_advs = []
    K, N = is_adv.shape                             # K = #epsilons, N = #images

    for i in range(N):                              # for each image in batch
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

    print(f"   GPU Mem In Adversarial Creation function: Alloc: {torch.cuda.memory_allocated(device)/(1024**2):.1f}MB, MaxAlloc: {torch.cuda.max_memory_allocated(device)/(1024**2):.1f}MB, Cached: {torch.cuda.memory_reserved(device)/(1024**2):.1f}MB")

    return selected_advs

if __name__ == '__main__':
    
    # Check if argument is provided
    if len(sys.argv) < 2:
        print("Error: Missing argument. Usage: python targeted_batch_wise_pipeline.py <index>")
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
    images = images[0:200]
    labels = labels[0:200]
    ids = torch.arange(len(images))
    dataset = TensorDataset(images, labels, ids)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    attack_to_epsilon = [{
    fb_att.L2ProjectedGradientDescentAttack(): np.linspace(0.5, 10, num=5),
    fb_att.L2FMNAttack(): np.linspace(0, 10, num=20),
    fb_att.L1FMNAttack(): np.linspace(2, 500, num=20),
    },                                                         
    {
    fb_att.LinfProjectedGradientDescentAttack(): np.linspace(0, 0.05, num=5),
    fb_att.LInfFMNAttack(): np.linspace(0, 0.5, num=20),
    },
    { 
    fb_att.LinfBasicIterativeAttack(): np.linspace(0, 0.1, num=15),
    fb_att.L2CarliniWagnerAttack(steps=1000): np.linspace(0, 10, num=10)
    }, 
    {
    fb_att.DDNAttack(): np.linspace(0, 10, num=20),
    fb_att.L2BasicIterativeAttack(): np.linspace(0, 15, num=30),
    } ]

    explanation_methods = {"GradCAM": GradCAM,  "GradCAMPlusPlus": GradCAMPlusPlus, "EigenCAM": EigenCAM, 
                           "EigenGradCAM": EigenGradCAM, "LayerCAM": LayerCAM, "KPCA_CAM": KPCA_CAM, 
                           "AblationCAM": AblationCAM, "FullGrad": FullGrad, "ScoreCAM": ScoreCAM}
    

    # give exact same saliency maps: "HiResCAM": HiResCAM, "GradCAMElementWise": GradCAMElementWise, "XGradCAM": XGradCAM
    all_attacks = attack_to_epsilon[attack_group_index]
    csv_file = "results/targeted/cam_comparison_metrics" + str(attack_group_index) + ".csv"
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
            adv_images = create_targeted_adversarials(batch_images, batch_labels, attack, epsilons)
            num_none = sum(adv is None for adv in adv_images)
            print(f"No adversarial in {num_none} cases for current batch.")
            num_adv_failed += num_none

            batch_images, batch_labels, batch_ids, adv_images = filter_adversarial_fails(batch_images, batch_labels, batch_ids, adv_images)
            print(batch_labels)
            if adv_images is None:
                continue

            for xAImethod_name, xAImethod in explanation_methods.items():
                print(f"Trying explanation method: {xAImethod}")
                if xAImethod_name == "FullGrad":
                    micro_batch_size = 8
                    create_explanations_micro_batch_wise(model, batch_images, batch_labels, xAImethod, attack, adv_images, csv_file, batch_ids, micro_batch_size, device)
                elif xAImethod_name == "ScoreCAM":
                    micro_batch_size = 2
                    create_explanations_micro_batch_wise(model, batch_images, batch_labels, xAImethod, attack, adv_images, csv_file, batch_ids, micro_batch_size, device)
                else:
                    micro_batch_size = 64
                    create_explanations_micro_batch_wise(model, batch_images, batch_labels, xAImethod, attack, adv_images, csv_file, batch_ids, micro_batch_size, device)


        print(f"Failed to create adversarial in {num_adv_failed} cases. For attack: {attack}. \n")