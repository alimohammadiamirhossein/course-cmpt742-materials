import cv2
import numpy as np
from scipy.sparse import csr_matrix, linalg


from align_target import align_target


def get_index_matrix(target_mask):
    # Get the index in the mask
    indices = np.column_stack(np.where(target_mask > 0))
    index_matrix = np.zeros(target_mask.shape, dtype=np.int32)
    for idx, (y, x) in enumerate(indices):
        index_matrix[y, x] = idx
    return index_matrix

def least_square_error(A, b, f, i, source_path):
    # Calculate least square error
    channel_map = {0: 'Blue', 1: 'Green', 2: 'Red'}
    error = np.linalg.norm(A @ f - b)
    source_name = source_path.replace('.jpg', '')
    print(f"{source_name} - {channel_map[i]} channel - Least square error: {error}")
    # Write error into a file
    with open('error_output.txt', 'a') as file:
        file.write(f"{source_name} - {channel_map[i]} channel - Least square error: {error}\n")
    return

def poisson_blend(source_image, target_image, target_mask, source_path):
    # Create an output image that initially just copies the target
    index_matrix = get_index_matrix(target_mask)
    output = np.copy(target_image)

    # Get indices of the non-zero values in the mask
    indices = np.column_stack(np.where(target_mask > 0))
    channels = target_image.shape[2]

    # Initialize A and b
    size = indices.shape[0]
    A = np.zeros((size, size))
    b = np.zeros((size, channels))

    # Directions of four neighbouring pixels
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for index, (y, x) in enumerate(indices):
        bool_neighbors = [target_mask[y + dy, x + dx] > 0 for dy, dx in directions]

        if all(bool_neighbors):
            A[index, index] = 4
            for dy, dx in directions:
                A[index, index_matrix[y + dy, x + dx]] = -1
                b[index, :] += source_image[y, x, :].astype(int) - source_image[y + dy, x + dx, :].astype(int)
        else:
            A[index, index] = 1
            b[index, :] += target_image[y, x, :]

    # Solve the matrix equation
    f = np.zeros((size, channels))
    for i in range(channels):  # Iterate over the three channels (RGB)
        f[:, i] = linalg.spsolve(csr_matrix(A), csr_matrix(b)[:, i])
        least_square_error(A, b[:, i], f[:, i], i, source_path)
    print(np.max(f), np.min(f))

    f= np.clip(f, 0, 255)

    for index, (y, x) in enumerate(indices):
       output[y, x, :] = abs(f[index, :])

    return output


if __name__ == '__main__':
    #Read source and target images
    #source_paths = ['source1.jpg', 'source2.jpg']
    source_paths = ['source1.jpg']
    target_path = 'target.jpg'
    target_image = cv2.imread(target_path)

    for source_path in source_paths:
        source_image = cv2.imread(source_path)

        # Align target image
        im_source, mask = align_target(source_image, target_image)

        # Save the source image
        output_filename = source_path.replace('.jpg', 'before_blended.jpg')
        cv2.imwrite(output_filename, im_source)

        # Poisson blend
        blended_image = poisson_blend(im_source, target_image, mask, source_path)

        # Save the blended image
        output_filename = source_path.replace('.jpg', '_blended.jpg')
        cv2.imwrite(output_filename, blended_image)

        # Display the blended image
        cv2.imshow('Blended Image', blended_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()