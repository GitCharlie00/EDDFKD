import glob
import os
import torch

from tqdm import tqdm

def evaluator(student,test_loader,val_criterion,history,device,log_interval):
    student.eval()
    val_correct_total = 0
    val_samples_total = 0
    test_bar = tqdm(test_loader, desc="Validation", leave=True)
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(test_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = student(images)
            loss = val_criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            val_correct_total += correct
            val_samples_total += labels.size(0)

            # Update history
            history["S_val_loss"].append(loss.item())
            history["S_val_accuracy"].append(accuracy)

            # Update progress bar
            if (batch_idx + 1) % log_interval == 0:
                test_bar.set_postfix(L_S=f"{loss.item():.4f}", acc_S=f"{accuracy:.4f}")

    return val_correct_total, val_samples_total

def save_best_model(args,teacher_name,student_name,student,epoch):
    if args.kdci:
        model_save_path = f"{args.save_path}{args.dataset}/{teacher_name.lower()}_{student_name.lower()}_{args.dataset.lower()}_{args.approach.lower()}_{args.method.lower()}_kdci_best_*"
        model_name_save_path = f"{args.save_path}{args.dataset}/{teacher_name.lower()}{student_name.lower()}_{args.dataset.lower()}_{args.approach.lower()}_{args.method.lower()}_kdci_best_{epoch+1}.pth"
    else:
        model_save_path = f"{args.save_path}{args.dataset}/{teacher_name.lower()}_{student_name.lower()}_{args.dataset.lower()}_{args.approach.lower()}_{args.method.lower()}_best_*"
        model_name_save_path = f"{args.save_path}{args.dataset}/{teacher_name.lower()}_{student_name.lower()}_{args.dataset.lower()}_{args.approach.lower()}_{args.method.lower()}_best_{epoch+1}.pth"

    for f in glob.glob(model_save_path):
        if os.path.isfile(f):
            os.remove(f)
            
    torch.save(student.state_dict(), model_name_save_path)
        
        
