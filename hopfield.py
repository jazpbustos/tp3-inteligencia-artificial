import numpy as np


# -------------------------
# Agregar ruido
# -------------------------
def agregar_ruido(vector, porcentaje):

    ruidoso = vector.copy()

    cantidad = int(len(vector) * porcentaje)

    indices = np.random.choice(
        len(vector),
        cantidad,
        replace=False
    )

    ruidoso[indices] *= -1

    return ruidoso


# -------------------------
# Recuperación Hopfield
# -------------------------
def recuperar(W, patron_inicial, iteraciones=10):

    estado = patron_inicial.copy()

    for _ in range(iteraciones):

        nuevo_estado = np.sign(W @ estado)

        nuevo_estado[nuevo_estado == 0] = 1

        estado = nuevo_estado

    return estado


# -------------------------
# Energía de Hopfield
# -------------------------
def energia(x, W):

    return -0.5 * x.T @ W @ x


# -------------------------
# Mostrar patrón
# -------------------------
def mostrar_patron(vector):

    matriz = vector.reshape(10, 10)

    for fila in matriz:
        print(
            "".join(
                "#" if pixel == 1 else "."
                for pixel in fila
            )
        )


# -------------------------
# Patrón original (anillo)
# -------------------------
ring = np.array([
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,0,1,1,1,1,0,0,0]
])

print("PATRÓN ORIGINAL\n")

for fila in ring:
    print("".join("#" if x else "." for x in fila))

# -------------------------
# Conversión a bipolar
# -------------------------
pattern = np.where(ring == 0, -1, 1)

x = pattern.flatten()

print("\nPrimeras 20 neuronas:")
print(x[:20])

# -------------------------
# Entrenamiento Hebb
# -------------------------
W = np.outer(x, x)

# Eliminar autoconexiones
np.fill_diagonal(W, 0)

print("\nTamaño de la matriz de pesos:")
print(W.shape)

# -------------------------
# Pruebas con distintos niveles de ruido
# -------------------------
niveles_ruido = [0.10, 0.30, 0.50]

for ruido in niveles_ruido:

    print("\n" + "=" * 50)
    print(f"RUIDO: {int(ruido * 100)}%")
    print("=" * 50)

    x_ruidoso = agregar_ruido(x, ruido)

    print("\nPatrón con ruido:\n")
    mostrar_patron(x_ruidoso)

    recuperado = recuperar(W, x_ruidoso)

    print("\nPatrón recuperado:\n")
    mostrar_patron(recuperado)

    print("\nEnergía final:")
    print(energia(recuperado, W))