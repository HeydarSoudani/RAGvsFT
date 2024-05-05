import os
from PIL import Image, ImageDraw, ImageFont

# === Fig 1 & 2 ===========
# def combine_images_with_titles(directory, output_path):
#     # List all PNG files in the directory
#     image_paths = [
#         # Figure 1
#         os.path.join(directory, 'bk_dist_popQA.png'),
#         os.path.join(directory, 'bk_dist_witQA.png'),
#         os.path.join(directory, 'bk_dist_EQ.png'),
        
#         # Figure 2
#         # os.path.join(directory, 'ret_recall_popQA.png'),
#         # os.path.join(directory, 'ret_recall_witQA.png'),
#         # os.path.join(directory, 'ret_recall_EQ.png'),
        
#     ]
    
#     # image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
#     images = [Image.open(path) for path in image_paths]
#     titles = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]  # Extracts the file names without extension
    
#     # Assume all images have the same height and calculate total width
#     height = max(img.height for img in images) + 60  # Adding space for titles
#     width = sum(img.width for img in images)
    
#     # Create a new image with a white background
#     combined_image = Image.new('RGB', (width, height), 'white')
#     draw = ImageDraw.Draw(combined_image)
    
#     # Load a font (this is using the default font, you may want to specify a path to a TTF font)
#     font = ImageFont.load_default()
    
#     # Position for the first image
#     x_offset = 0
    
#     # Draw images and titles
#     for title, img in zip(titles, images):
#         # Draw title
#         title_height = 30  # Space allocated for the title
#         # draw.text((x_offset + 10, 10), title, font=font, fill='black')
        
#         # Draw image
#         combined_image.paste(img, (x_offset, title_height))
#         x_offset += img.width
    
#     # Save the combined image
#     combined_image.save(output_path)

# image_directory = 'analysis/images_results/1_popularity_per_bucket'  # 1_popularity_per_bucket, 2_ret_buckets
# output_image_path = f'{image_directory}/combined_image.pdf'

# combine_images_with_titles(image_directory, output_image_path)


# === Fig 3 ===============
def combine_labeled_images(directory, output_path):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    details = [(int(f.split('_')[0]), f.split('_')[1], f.split('_')[2].replace('.png', ''), f) for f in files]
    
    # models = sorted(set(detail[1] for detail in details))
    # datasets = sorted(set(detail[2] for detail in details))
    models = ['flant5sm', 'flant5bs', 'flant5lg', 'flant5xl', 'flant5xxl']
    # models = ['tinyllama', 'stablelm2', 'minicpm', 'llama2', 'mistral', 'zephyr']
    datasets = ['popqa', 'witqa', 'eq']
    
    num_rows = len(models)
    num_cols = len(datasets)
    
    image_dict = {}
    for row, model, dataset, filename in details:
        if model not in image_dict:
            image_dict[model] = {}
        image_dict[model][dataset] = Image.open(os.path.join(directory, filename))
    
    first_model = next(iter(image_dict))
    first_dataset = next(iter(image_dict[first_model]))
    img_width, img_height = image_dict[first_model][first_dataset].size

    # Calculate margins for labels
    label_height = 50  # height for column labels
    label_width = 100  # width for row labels
    total_width = img_width * num_cols + label_width
    total_height = img_height * num_rows + label_height

    # Create a new image with a white background
    combined_img = Image.new('RGB', (total_width, total_height), 'white')
    
    # Prepare to draw text
    draw = ImageDraw.Draw(combined_img)
    # font = ImageFont.load_default()  # Using the default font
    font_size = 36  # Font size for labels
    font = ImageFont.truetype("Arial.ttf", font_size)
    # font = ImageFont.truetype("Arial_Bold.ttf", font_size)

    # Paste images and draw labels
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            x_offset = j * img_width + label_width
            y_offset = i * img_height + label_height
            img = image_dict[model][dataset]
            combined_img.paste(img, (x_offset, y_offset))
            if i == 0:  # Draw column labels on the first row
                draw.text((x_offset + img_width//2 - 10*len(dataset), 10), dataset, font=font, fill="navy")
        # Draw row labels on the first column
        draw.text((10, y_offset + img_height//2 - 10), model, font=font, fill="navy")

    # Save the combined image
    combined_img.save(output_path)
    


image_directory = 'analysis/images_results/3_answer_generator/flans'  # flans or llms
output_image_path = f'{image_directory}/combined_image.pdf'

combine_labeled_images(image_directory, output_image_path)








