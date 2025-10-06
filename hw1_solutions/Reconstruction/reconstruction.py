import numpy as np
import cv2
from scipy.sparse import csr_matrix, linalg
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

def construct_Ab(S, c):
    h, w = S.shape
    A = lil_matrix((h * w, h * w))
    b = np.zeros(h * w)

    # Corner coordinates.
    corner_coords = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

    for y in range(h):
        for x in range(w):
            idx = y * w + x

            # Directions of four neighbouring pixels
            neighbour_coords = {
                "c": (y, x),
                "l": (y, x - 1),
                "r": (y, x + 1),
                "d": (y + 1, x),
                "u": (y - 1, x)
            }

            # Four corners condition
            if (y, x) in corner_coords:
                index = corner_coords.index((y, x))
                A[idx, idx] = 1
                b[idx] = c[index]
                #b[idx] = S[y, x]
                #print(S[y, x], index)

            # Top or bottom edge
            elif y == 0 or y == h - 1:
                A[idx, idx] = 2
                A[idx, idx + 1] = A[idx, idx - 1] = -1
                b[idx] = 2 * S[neighbour_coords['c']] - S[neighbour_coords['l']] - S[neighbour_coords['r']]

            # Left and right edge
            elif x == 0 or x == w - 1:
                A[idx, idx] = 2
                A[idx, idx + w] = A[idx, idx - w] = -1
                b[idx] = 2 * S[neighbour_coords['c']] - S[neighbour_coords['u']] - S[neighbour_coords['d']]

            # Other pixels
            else:
                A[idx, idx] = 4
                A[idx, idx + 1] = A[idx, idx - 1] = A[idx, idx + w] = A[idx, idx - w] = -1
                b[idx] = 4 * S[neighbour_coords['c']] - S[neighbour_coords['l']] - S[neighbour_coords['r']] - S[neighbour_coords['u']] - S[neighbour_coords['d']]

    return A, b

def least_square_error(A, b, V, img_path):
    # Calculate and write the least square error
    error = np.linalg.norm(A @ V - b)
    img_name = img_path.replace('.jpg', '')
    print(f"{img_name} channel - Least square error: {error}")
    # Write the square error into the file
    with open('error_output.txt', 'a') as file:
        file.write(f"{img_name}  channel - Least square error: {error}\n")
    return

def reconstruct_image2(S, c, img_path):
    # Solve the matrix equation
    A, b = construct_Ab(S, c)
    V = linalg.spsolve(csr_matrix(A), csr_matrix(b).T)
    print(V.min(), V.max())
    least_square_error(A, b, V, img_path)
    return V.reshape(S.shape)


def reconstruct_image(S, c, img_path):
    n_rows, n_cols = S.shape
    print(S.dtype)
    S = S.astype(np.float64)
    k = n_rows * n_cols
    A = lil_matrix((k, k))
    b = np.zeros(k)

    # Define the corner positions and their corresponding values
    corners = {(0, 0): c[0], (0, n_cols - 1): c[1], (n_rows - 1, 0): c[2], (n_rows - 1, n_cols - 1): c[3]}
    
    for x in range(n_rows):
        for y in range(n_cols):
            i = x * n_cols + y
            coeff = 0
            rhs = 0

            A[i, i] = 0
            # Handle corners explicitly
            if (x, y) in corners:
                A[i, i] = 1  # Set the corner as a fixed point
                b[i] = corners[(x, y)]
            else:
                # Handle interior pixels using finite differences
                if n_rows - 1 > x > 0:
                    A[i, i] += 2
                    A[i, (x-1) * n_cols + y] = -1
                    A[i, (x+1) * n_cols + y] = -1
                    rhs += (S[x, y] - S[x - 1, y])
                    rhs += (S[x, y] - S[x + 1, y])
                    coeff += 1
                if 0 <  y < n_cols - 1:
                    A[i, i] += 2
                    A[i, x * n_cols + (y-1)] = -1
                    A[i, x * n_cols + (y+1)] = -1
                    rhs += (S[x, y] - S[x, y - 1])
                    rhs += (S[x, y] - S[x, y + 1])
                    coeff += 1
                if coeff > 0:
                    b[i] = rhs
                # Handle any case where coeff == 0 (not likely here)

    print(A.shape, b.shape)

    # Solve the system using LSQR
    # v, istop, itn, normr = lsqr(A.tocsr(), b)[:4]
    # print(istop, itn, normr, v.shape, v.max(), v.min())
    v = linalg.spsolve(csr_matrix(A), csr_matrix(b).T)

    # Calculate the least squares error
    ls_error = np.linalg.norm(A.dot(v) - b)
    print("LS Error:", ls_error)

    # Rescale the result to the range [0, 1]
    print(v.min(), v.max(), 'minmax')
    v = (v - v.min()) / (v.max() - v.min() ) * 255.

    # Reshape the solution vector back into the image
    V = v[:k].reshape((n_rows, n_cols))
    
    return V




if __name__ == '__main__':
    root_path = "./"
    img_names = ["large", "large1", "target", "target1"]
    c = [150, 150, 150, 150]

    for name in img_names:
        # Read the image and convert it to grayscale
        img_path = root_path + name + ".jpg"
        S = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

        # Reconstruct the image
        c = [S[0, 0], S[0, -1], S[-1, 0], S[-1, -1]]
        V = reconstruct_image(S, c, img_path)

        # Combine S and V side-by-side
        combined = np.hstack((S, V))

        # Display the combined image
        # cv2.imshow(f'{name} - Source and Reconstructed Image', combined.astype(np.float64) / 255.)

        # Save the combined image
        save_path = root_path + name + "_adjust1.png"
        cv2.imwrite(save_path, combined)  # Multiply by 255 to save as an 8-bit image

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
