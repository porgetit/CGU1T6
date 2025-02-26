import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def generate_matrix():
    A = np.random.randint(1, 16, (3, 5))
    return A

def matrix_operations(A):
    sumaA = np.sum(A)
    productoA = np.prod(A)
    mediaA = np.mean(A)
    a = A[1, 1:3]
    B = np.copy(A[A > 7])
    return sumaA, productoA, mediaA, a, B

def generate_random_matrix():
    C = np.random.randint(0, 10, (3, 3))
    detC = np.linalg.det(C)
    invC = np.linalg.inv(C)
    return C, detC, invC

def generate_large_random_array():
    D = np.random.randint(0, 100, (1, 100))
    maxD = np.max(D)
    minD = np.min(D)
    meanD = np.mean(D)
    stdD = np.std(D)
    return D, maxD, minD, meanD, stdD

def plot_sine_cosine():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label='sin(x)')
    plt.plot(x, y2, label='cos(x)')
    plt.legend()
    plt.title('Funciones Seno y Coseno')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_scatter(D):
    plt.scatter(np.arange(D.size), D)
    plt.title('Diagrama de Dispersión de D')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.show()

def plot_histogram(D):
    plt.hist(D.flatten(), bins=10)
    plt.title('Histograma de D')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.show()

def plot_images():
    img = mpimg.imread('image.jpg')
    img_gray = np.mean(img, axis=2)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title('Imagen en Escala de Grises')
    plt.imshow(img_gray, cmap='gray')

    plt.show()

def main():
    # Generar y operar en la matriz A
    A = generate_matrix()
    sumaA, productoA, mediaA, a, B = matrix_operations(A)
    print("Matriz A:\n", A)
    print("Suma de A:", sumaA)
    print("Producto de A:", productoA)
    print("Media de A:", mediaA)
    print("Submatriz de A:", a)
    print("Elementos de A > 7:", B)

    # Generar y operar en la matriz C
    C, detC, invC = generate_random_matrix()
    print("\nMatriz C:\n", C)
    print("Determinante de C:", detC)
    print("Inversa de C:\n", invC)

    # Generar y operar en el array D
    D, maxD, minD, meanD, stdD = generate_large_random_array()
    print("\nArray D:\n", D)
    print("Máximo de D:", maxD)
    print("Mínimo de D:", minD)
    print("Media de D:", meanD)
    print("Desviación Estándar de D:", stdD)

    # Graficar funciones seno y coseno
    plot_sine_cosine()

    # Graficar diagrama de dispersión de D
    plot_scatter(D)

    # Graficar histograma de D
    plot_histogram(D)

    # Graficar imágenes original y en escala de grises
    plot_images()

if __name__ == "__main__":
    main()
