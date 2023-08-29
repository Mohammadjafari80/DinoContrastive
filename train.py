import argparse
import json
import pathlib
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from evaluate import compute_knn
from utils import DataAugmentation, Head, Loss, MultiCropWrapper, clip_gradients
from data import get_train_dataset, get_test_dataset

def main():
    parser = argparse.ArgumentParser(
        "DINO training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=("cifar10", "cifar100", 'mnist', 'fashion', 'mvtec', 'svhn', 'cub'))
    parser.add_argument(
        "-d", "--device", type=str, choices=("cpu", "cuda"), default="cuda"
    )
    parser.add_argument("--label", type=int, help="The normal class", nargs="+")
    parser.add_argument("-l", "--logging-freq", type=int, default=5)
    parser.add_argument("--momentum-teacher", type=float, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-e", "--n-epochs", type=int, default=100)
    parser.add_argument("-o", "--out-dim", type=int, default=16)
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14")
    parser.add_argument("--repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--path", type=str, default="~/data/")
    parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)

    parser.add_argument("--feature-dim", type=int, default=384)
    
    args = parser.parse_args()
    print(vars(args))
    
    # Parameters
    dim = args.feature_dim

    logging_path = pathlib.Path(args.tensorboard_dir)
    device = torch.device(args.device)

    n_workers = 2

    dataset_train_plain,  dataset_train_aug = get_train_dataset(args.dataset, args.label, args.path, 'backbone')
    dataset_val_plain = get_test_dataset(args.dataset, args.label, args.path, 'backbone')


    data_loader_train_aug = DataLoader(
        dataset_train_aug,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    data_loader_train_plain = DataLoader(
        dataset_train_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
    data_loader_val_plain = DataLoader(
        dataset_val_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )

    # Logging
    writer = SummaryWriter(logging_path)
    writer.add_text("arguments", json.dumps(vars(args)))

    # Neural network related
    student_vit =  torch.hub.load(args.repo, args.dino_model)
    teacher_vit =  torch.hub.load(args.repo, args.dino_model)

    student = MultiCropWrapper(
        student_vit,
        Head(
            dim,
            args.out_dim,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    
    teacher = MultiCropWrapper(teacher_vit, Head(dim, args.out_dim))
    student, teacher = student.to(device), teacher.to(device)

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    for p in student.backbone.parameters():
        p.requires_grad = False
        
    # Loss related
    loss_inst = Loss(
        args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    ).to(device)
    
    lr = 0.0005 * args.batch_size / 256
    
    optimizer = torch.optim.AdamW(
        student.new_head.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    n_batches = len(dataset_train_aug) // args.batch_size

    for e in range(args.n_epochs):
        
        if e % args.logging_freq == 0:
            student.eval()

            # KNN
            current_auc = compute_knn(
                student,
                data_loader_train_plain,
                data_loader_val_plain,
                device
            )
            print(f"Epoch {e}, KNN AUROC: {current_auc}")

            student.train()

        # Initialize tqdm inline within the outer for loop, after the test section
        with tqdm(
            enumerate(data_loader_train_aug), 
            total=n_batches, 
            desc="Training",
            dynamic_ncols=True
        ) as pbar:

            for i, (images, _) in pbar:
                images = [img.to(device) for img in images]

                teacher_output = teacher(images[:2])
                student_output = student(images)

                loss = loss_inst(student_output, teacher_output)

                optimizer.zero_grad()
                loss.backward()
                clip_gradients(student, args.clip_grad)
                optimizer.step()

                with torch.no_grad():
                    for student_ps, teacher_ps in zip(
                        student.parameters(), teacher.parameters()
                    ):
                        teacher_ps.data.mul_(args.momentum_teacher)
                        teacher_ps.data.add_(
                            (1 - args.momentum_teacher) * student_ps.detach().data
                        )

                # Update tqdm description to show the current loss
                pbar.set_description(f"Training (loss: {loss.item():.4f})")

    # for e in range(args.n_epochs):
    #     for i, (images, _) in tqdm.tqdm(
    #         enumerate(data_loader_train_aug), total=n_batches
    #     ):
    #         if n_steps % args.logging_freq == 0:
    #             student.eval()
                
    #             # KNN
    #             current_auc = compute_knn(
    #                 student,
    #                 data_loader_train_plain,
    #                 data_loader_val_plain,
    #                 device
    #             )
    #             print("KNN AUROC: ", current_auc)
                
    #             student.train()

    #         images = [img.to(device) for img in images]

    #         teacher_output = teacher(images[:2])
    #         student_output = student(images)

    #         loss = loss_inst(student_output, teacher_output)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         clip_gradients(student, args.clip_grad)
    #         optimizer.step()

    #         with torch.no_grad():
    #             for student_ps, teacher_ps in zip(
    #                 student.parameters(), teacher.parameters()
    #             ):
    #                 teacher_ps.data.mul_(args.momentum_teacher)
    #                 teacher_ps.data.add_(
    #                     (1 - args.momentum_teacher) * student_ps.detach().data
    #                 )

    #         n_steps += 1


if __name__ == "__main__":
    main()