from PIL import Image, ImageDraw

def draw_square(image_path, cell_centroids, box_size = 20):
    half_box_size = box_size // 2
    image_with_cell_boxes = Image.open(image_path)
    draw = ImageDraw.Draw(image_with_cell_boxes)

    # Extract each centroid in the list, convert to integers, and draw the square
    for x, y in cell_centroids:
        x, y = int(x), int(y)
        box = [x-half_box_size, y-half_box_size, x+half_box_size, y+half_box_size]  # Creates a box with length 20 centered at (x, y)
        draw.rectangle(box, outline='red', width=2)
    return image_with_cell_boxes


