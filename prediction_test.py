"""
Test script for evaluating model predictions on test dataset.
"""
import os
from colorama import Fore, Style
from predictions import predict_body_part, predict_fracture
from config import TEST_DIR, BODY_PARTS


# load images to predict from paths
#               ....                       /    elbow1.jpg
#               Hand          fractured  --   elbow2.png
#           /                /             \    .....
#   test   -   Elbow  ------
#           \                \         /        elbow1.png
#               Shoulder        normal --       elbow2.jpg
#               ....                   \
#
def load_path(path):
    """
    Load test images from directory structure.
    
    Args:
        path: Path object to test directory
        
    Returns:
        List of image dictionaries
    """
    dataset = []
    if not path.exists():
        return dataset
    
    for body in os.listdir(path):
        body_part = body
        path_p = path / str(body)
        if not path_p.is_dir():
            continue
        
        for lab in os.listdir(path_p):
            label = lab
            path_l = path_p / str(lab)
            if not path_l.is_dir():
                continue
            
            for img in os.listdir(path_l):
                img_path = path_l / str(img)
                if img_path.is_file():
                    dataset.append({
                        'body_part': body_part,
                        'label': label,
                        'image_path': str(img_path),
                        'image_name': img
                    })
    return dataset


categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']


def reportPredict(dataset):
    """
    Generate prediction report for test dataset.
    
    Args:
        dataset: List of image dictionaries with 'body_part', 'label', 'image_path', 'image_name'
    """
    if not dataset:
        print(Fore.RED + "No test images found!")
        return
    
    total_count = len(dataset)
    part_count = 0
    status_count = 0
    part_confidences = []
    fracture_confidences = []

    print(Fore.YELLOW + Style.BRIGHT +
          '{0: <28}'.format('Name') +
          '{0: <14}'.format('Part') +
          '{0: <20}'.format('Predicted Part') +
          '{0: <12}'.format('Part Conf') +
          '{0: <20}'.format('Status') +
          '{0: <20}'.format('Predicted Status') +
          '{0: <12}'.format('Fracture Conf'))
    print(Style.RESET_ALL)
    
    for i, img in enumerate(dataset):
        try:
            # Predict body part
            body_part_predict, part_conf = predict_body_part(img['image_path'])
            part_confidences.append(part_conf)
            
            # Predict fracture
            fracture_predict, fracture_conf = predict_fracture(img['image_path'], body_part_predict)
            fracture_confidences.append(fracture_conf)
            
            # Check accuracy
            part_correct = img['body_part'] == body_part_predict
            status_correct = img['label'] == fracture_predict
            
            if part_correct:
                part_count += 1
            if status_correct:
                status_count += 1
                color = Fore.GREEN
            else:
                color = Fore.RED
            
            print(color +
                  '{0: <28}'.format(img['image_name'][:27]) +
                  '{0: <14}'.format(img['body_part']) +
                  '{0: <20}'.format(body_part_predict) +
                  '{0: <12}'.format(f"{part_conf:.1%}") +
                  '{0: <20}'.format(img['label']) +
                  '{0: <20}'.format(fracture_predict) +
                  '{0: <12}'.format(f"{fracture_conf:.1%}"))
            
        except Exception as e:
            print(Fore.RED + f"Error processing {img['image_name']}: {e}")
            continue

    # Print summary
    print(Style.RESET_ALL)
    print(Fore.BLUE + Style.BRIGHT + "\n" + "="*60)
    print("SUMMARY")
    print("="*60 + Style.RESET_ALL)
    print(Fore.BLUE + f'Total images: {total_count}')
    print(Fore.BLUE + f'Body Part Accuracy: {part_count}/{total_count} = {part_count/total_count*100:.2f}%')
    print(Fore.BLUE + f'Fracture Detection Accuracy: {status_count}/{total_count} = {status_count/total_count*100:.2f}%')
    
    if part_confidences:
        avg_part_conf = sum(part_confidences) / len(part_confidences)
        print(Fore.BLUE + f'Average Body Part Confidence: {avg_part_conf:.1%}')
    
    if fracture_confidences:
        avg_fracture_conf = sum(fracture_confidences) / len(fracture_confidences)
        print(Fore.BLUE + f'Average Fracture Detection Confidence: {avg_fracture_conf:.1%}')
    
    print(Style.RESET_ALL)


if __name__ == "__main__":
    if not TEST_DIR.exists():
        print(Fore.RED + f"Test directory not found: {TEST_DIR}")
        exit(1)
    
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*60)
    print("BONE FRACTURE DETECTION - TEST EVALUATION")
    print("="*60 + Style.RESET_ALL + "\n")
    
    dataset = load_path(TEST_DIR)
    
    if not dataset:
        print(Fore.RED + "No test images found in test directory!")
        exit(1)
    
    print(f"Found {len(dataset)} test images\n")
    reportPredict(dataset)
