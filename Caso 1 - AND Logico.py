import numpy as np

class Perceptron:
    def __init__(self, tasa_aprendizaje=0.1, iteraciones=10):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None

    def funcion_activacion(self, x):
        return 1 if x >= 0 else 0

    def entrenar(self, entradas, salidas):
        n_muestras, n_caracteristicas = entradas.shape
        self.pesos = np.zeros(n_caracteristicas)
        self.sesgo = 0

        for _ in range(self.iteraciones):
            for indice, entrada in enumerate(entradas):
                salida_lineal = np.dot(entrada, self.pesos) + self.sesgo
                salida_predicha = self.funcion_activacion(salida_lineal)

                actualizacion = self.tasa_aprendizaje * (salidas[indice] - salida_predicha)
                self.pesos += actualizacion * entrada
                self.sesgo += actualizacion

    def predecir(self, entradas):
        salida_lineal = np.dot(entradas, self.pesos) + self.sesgo
        return np.array([self.funcion_activacion(x) for x in salida_lineal])

# Datos para compuerta lógica AND
entradas_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salidas_and = np.array([0, 0, 0, 1])

# Crear el perceptrón
perceptron_and = Perceptron()

# Entrenar el perceptrón con los datos AND
perceptron_and.entrenar(entradas_and, salidas_and)

# Evaluar el perceptrón
predicciones = perceptron_and.predecir(entradas_and)
precision = np.mean(predicciones == salidas_and) * 100

print("Predicciones AND:", predicciones)
print(f"Precisión del modelo: {precision:.2f}%")
print(f"Pesos finales: {perceptron_and.pesos}")
print(f"Sesgo final: {perceptron_and.sesgo}")

# Permitir al usuario insertar valores nuevos
print("\n--- Inserta valores para probar el perceptrón entrenado ---")
print("Escribe 'salir' para terminar.\n")

while True:
    entrada_usuario = input("Introduce dos valores separados por espacio (ejemplo: 1 0): ")
    if entrada_usuario.lower() == 'salir':
        print("Finalizando ...")
        break
    try:
        valores = list(map(int, entrada_usuario.strip().split()))
        if len(valores) != 2:
            print("Por favor introduce exactamente dos valores (0 o 1).")
            continue
        valores_array = np.array(valores)
        prediccion = perceptron_and.predecir(valores_array.reshape(1, -1))
        print(f"Predicción del perceptrón: {prediccion[0]}\n")
    except ValueError:
        print("Entrada inválida. Asegúrate de escribir números separados por espacio.\n")
