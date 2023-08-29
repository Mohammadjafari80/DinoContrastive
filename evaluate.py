import numpy as np
import torch
import faiss
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def compute_knn(backbone, data_loader_train, data_loader_val, device):
    """Get CLS embeddings and use KNN distance on them.

    We load all embeddings in memory and use sklearn. Should
    be doable.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer whose head is just an identity
        mapping.

    data_loader_train, data_loader_val : torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any
        augmentations. Just casting to tensor and then normalizing.

    Returns
    -------
    val_auroc : float
        Validation auroc.
    """
    train_feature_space_pretrained = []
    train_feature_space_finetuned = []
    
    print("Trainset feature extracting...")
    
    with torch.no_grad():
        for index, (imgs, _) in enumerate(data_loader_train):
            imgs = imgs.to(device)
            features = backbone(imgs, True)
            train_feature_space_pretrained.append(features[0].detach().cpu())
            train_feature_space_finetuned.append(features[1].detach().cpu())
            
        train_feature_space_pretrained = (
            torch.cat(train_feature_space_pretrained, dim=0).contiguous().cpu().numpy()
        )
        
        train_feature_space_finetuned = (
            torch.cat(train_feature_space_finetuned, dim=0).contiguous().cpu().numpy()
        )
        
    test_feature_space_pretrained = []
    test_feature_space_finetuned = []
    test_labels = []
    
    print("Testset feature extracting...")
    
    with torch.no_grad():
        for index, (imgs, labels) in enumerate(data_loader_val):
            imgs = imgs.to(device)
            features = backbone(imgs, True)
            test_feature_space_pretrained.append(features[0].detach().cpu())
            test_feature_space_finetuned.append(features[1].detach().cpu())
            test_labels.append(labels)
            
        test_feature_space_pretrained = (
            torch.cat(test_feature_space_pretrained, dim=0).contiguous().cpu().numpy()
        )
        
        test_feature_space_finetuned = (
            torch.cat(test_feature_space_finetuned, dim=0).contiguous().cpu().numpy()
        )
        
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances_pretrained = knn_score(train_feature_space_pretrained, test_feature_space_pretrained)
    distances_finetuned = knn_score(train_feature_space_finetuned, test_feature_space_finetuned)
    distances_commbined = (distances_pretrained - np.min(distances_pretrained)) / (np.max(distances_pretrained) - np.min(distances_pretrained)) + (distances_finetuned - np.min(distances_finetuned)) / (np.max(distances_finetuned) - np.min(distances_finetuned))
    
    auc_pretrained = roc_auc_score(test_labels, distances_pretrained)
    auc_finetuned = roc_auc_score(test_labels, distances_finetuned)
    auc_combined = roc_auc_score(test_labels, distances_commbined)

    return auc_pretrained, auc_finetuned, auc_combined