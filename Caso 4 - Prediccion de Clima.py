import numpy as np

class Perceptron:
    def __init__(self, tasa_aprendizaje=0.01, iteraciones=100):
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

# Datos para predicción del clima (temperatura, humedad, nubosidad)
entradas_clima = np.array([
    [25, 70, 1],  # Lluvia
    [30, 50, 0],   # Soleado
    [15, 80, 1],   # Lluvia
    [28, 40, 0],   # Soleado
    [20, 90, 1],   # Lluvia
    [32, 35, 0]    # Soleado
])
salidas_clima = np.array([1, 0, 1, 0, 1, 0])  # 1 = Lluvia, 0 = Soleado

# Normalización
entradas_clima = entradas_clima / np.max(entradas_clima, axis=0)

# Crear y entrenar el perceptrón
perceptron_clima = Perceptron(tasa_aprendizaje=0.01, iteraciones=100)
perceptron_clima.entrenar(entradas_clima, salidas_clima)

# Evaluar
predicciones = perceptron_clima.predecir(entradas_clima)
precision = np.mean(predicciones == salidas_clima) * 100

print("\n--- Caso 4: Predicción del Clima ---")
print("Predicciones Clima:", predicciones)
print(f"Precisión: {precision:.2f}%")
print(f"Pesos finales: {perceptron_clima.pesos}")
print(f"Sesgo final: {perceptron_clima.sesgo}")

# Interacción con el usuario
print("\n--- Prueba el predictor de Clima ---")
print("Introduce 3 valores separados por espacio (temperatura, humedad, nubosidad)")
print("Nubosidad: 1 = nublado, 0 = despejado")
print("Ejemplo: 28 60 0")
print("Escribe 'salir' para terminar.\n")

while True:
    entrada_usuario = input("Introduce los valores: ")
    if entrada_usuario.lower() == 'salir':
        print("Finalizando ...")
        break
    try:
        valores = list(map(float, entrada_usuario.strip().split()))
        if len(valores) != 3:
            print("Por favor introduce exactamente 3 valores numéricos.")
            continue
        # Normalizar los valores de entrada
        max_vals = np.max(entradas_clima, axis=0)
        valores_norm = np.array(valores) / max_vals
        prediccion = perceptron_clima.predecir(valores_norm.reshape(1, -1))
        resultado = "LLUVIA" if prediccion[0] == 1 else "SOLEADO"
        print(f"\nDatos ingresados:")
        print(f"Temperatura: {valores[0]}°C, Humedad: {valores[1]}%, Nubosidad: {'Nublado' if valores[2] == 1 else 'Despejado'}")
        print(f"Predicción: {resultado}\n")
    except ValueError:
        print("Entrada inválida. Asegúrate de escribir números separados por espacio.\n")