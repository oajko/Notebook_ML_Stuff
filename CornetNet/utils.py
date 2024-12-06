import matplotlib.pyplot as plt
import matplotlib.patches as patch

# img path. json path
def display_im(im_path, json_data):
    file_name = im_path.split("\\")[-1]
    im_id = [i['id'] for i in json_data['images'] if i['file_name'] == file_name]
    box_locations = [i for i in json_data['annotations'] if i['image_id'] == im_id[0]]

    cat_mapper = {i['id']: i['name'] for i in  json_data['categories']}

    im = plt.imread(im_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(im)
    for box in box_locations:
        bbox = box['bbox']
        rect = patch.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec = 'green', lw = 2, fill = None)
        plt.gca().add_patch(rect)
        plt.text(bbox[0], bbox[1], cat_mapper[box['category_id']])
    plt.tight_layout()
    plt.show()