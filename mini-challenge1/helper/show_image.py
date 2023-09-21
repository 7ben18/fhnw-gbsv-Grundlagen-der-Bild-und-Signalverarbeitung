import matplotlib.pyplot as plt

def show_image(images_list, image_index):
    """
    Display the image at the specified index from the 'images' list.

    Parameters:
    - images_list (list): The list of images to display (as PIL image objects
    - image_index (int): The index of the image to display.

    """
    if 0 <= image_index < len(images_list):
        plt.imshow(images_list[image_index])
        plt.axis("off")
        plt.show()
    else:
        print("Invalid image index. Please provide a valid index.")
