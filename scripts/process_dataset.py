import os
import multiprocessing
from PIL import Image
import numpy as np
import argparse
import sys

def calculate_new_size(image_path, target_size):
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height

    if width < height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    return new_width, new_height

# NOTE: 因为多进程进度显示不准确，以实际保存的图片数量为准；令外参考报错信息
def resize_image(image_path, save_path, target_size, processed_count, total_to_process, save_mode='RGB'):
    try:
        new_width, new_height = calculate_new_size(image_path, target_size)
        img = Image.open(image_path).convert(save_mode)
        img = img.resize((new_width, new_height), Image.BILINEAR)
        # if save_mode in ['1', 'L', 'P']:
        #     image_array = np.array(img)
        #     # 处理像素值，将 [0, 127] 范围的像素置零，[128, 255] 范围内的像素置255
        #     image_array[image_array <= 127] = 0
        #     image_array[image_array >= 128] = 255
        #     # 将处理后的 NumPy 数组转换回 PIL 图像
        #     img = Image.fromarray(image_array)
        # img=img.convert(save_mode)
        img.save(save_path)
        processed_count.value += 1
        sys.stdout.write(f"\rProcessed: [{processed_count.value}/{total_to_process}]")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error processing {image_path}: {e}\n")

def process_images(image_dir, save_dir, target_size, num_processes, processed_count, save_mode):
    import time
    start_time=time.time()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    total_to_process = len(image_paths)

    pool = multiprocessing.Pool(processes=num_processes)

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, filename)
        pool.apply_async(resize_image, args=(image_path, save_path, target_size, processed_count, total_to_process, save_mode))

    pool.close()
    pool.join()
    
    end_time=time.time()
    print(f"\ntotal time consumed: {end_time-start_time:.3f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch resize images using multiple processes.")
    # parser.add_argument("image_dir", type=str, help="Path to the image folder")
    # parser.add_argument("save_dir", type=str, help="Path to the save folder")
    # parser.add_argument("--target_size", type=int, default=300, help="Target size of the shorter side")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of CPU cores to use")
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    
    data_root = "data/Image_Harmonization_Dataset"
    ds_list=['Hday2night','HCOCO','HFlickr','HAdobe5k']
    res_list=[256,512,1024]
    
    for res in res_list:
        for ds in ds_list:
            print(f"process dataset {ds} to {res}px")
            image_dir = os.path.join(data_root, f"{ds}_original")
            save_dir = os.path.join(data_root, f"{ds}_{res}px")
            
            processed_count = manager.Value('i', 0)
            process_images(os.path.join(image_dir,"masks"), os.path.join(save_dir, "masks"), res, args.num_processes, processed_count, "1")
            
            processed_count = manager.Value('i', 0)
            process_images(os.path.join(image_dir,"real_images"), os.path.join(save_dir, "real_images"), res, args.num_processes, processed_count, "RGB")
            
            processed_count = manager.Value('i', 0)
            process_images(os.path.join(image_dir,"composite_images"), os.path.join(save_dir, "composite_images"), res, args.num_processes, processed_count, "RGB")
            
            print("\nAll images processed successfully.")