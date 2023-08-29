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
    train_feature_space = []
    
    with torch.no_grad():
        for imgs, _ in tqdm(data_loader_train, desc="Train set feature extracting"):
            imgs = imgs.to(device)
            features = backbone(imgs)
            train_feature_space.append(features.detach().cpu())
        train_feature_space = (
            torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        )
        
    test_feature_space = []
    test_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader_val, desc="Test set feature extracting"):
            imgs = imgs.to(device)
            features = backbone(imgs)
            test_feature_space.append(features.detach().cpu())
            test_labels.append(labels)
        test_feature_space = (
            torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        )
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc