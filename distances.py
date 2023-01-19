from scipy.linalg import sqrtm
import numpy as np

def frechet_distance(gauss_x, gauss_y):
    mu_x, sigma_x = gauss_x
    mu_y, sigma_y = gauss_y
    # Formula from Dowson, D. C; Landau, B. V (1 September 1982).
    # "The Fréchet distance between multivariate normal distributions".
    # Journal of Multivariate Analysis. 12 (3): 450–455.
    # https://www.sciencedirect.com/science/article/pii/0047259X8290077X?via%3Dihub
    mean_diff = mu_x - mu_y
    matrix = sigma_x + sigma_y - 2 * sqrtm(sigma_x @ sigma_y)
    frechet_dist_sqr = np.dot(mean_diff, mean_diff) + np.trace(matrix)
    return np.sqrt(frechet_dist_sqr)