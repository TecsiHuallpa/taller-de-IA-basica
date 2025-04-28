import random


class PerceptronSpam:
    def __init__(self, max_length=100):
        """
        Inicializa el perceptrón para mensajes de hasta max_length caracteres.
        Cada carácter son 8 bits, así que tendremos max_length*8 entradas.
        """
        self.max_length = max_length
        self.input_size = max_length * 8  # Cada carácter ASCII son 8 bits
        self.weights = [random.uniform(-1, 1) for _ in range(self.input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01

    def texto_a_binario(self, mensaje):
        """
        Convierte un mensaje de texto a una lista de bits (binario).
        Rellena con ceros si el mensaje es más corto que max_length.
        """
        # Convertir cada carácter a su valor ASCII y luego a binario de 8 bits
        bits = []
        for char in mensaje[:self.max_length]:
            ascii_val = ord(char)
            char_bits = [int(b) for b in f"{ascii_val:08b}"]
            bits.extend(char_bits)

        # Rellenar con ceros si el mensaje es más corto
        bits += [0] * (self.input_size - len(bits))
        return bits

    def activacion(self, x):
        """Función de activación escalón (step function)"""
        return 1 if x >= 0 else 0

    def predecir(self, mensaje):
        """Predice si el mensaje es spam (1) o no (0)"""
        # Convertir texto a binario
        inputs = self.texto_a_binario(mensaje)

        # Calcular suma ponderada
        z = self.bias
        for i in range(self.input_size):
            z += self.weights[i] * inputs[i]

        # Aplicar función de activación
        return self.activacion(z)

    def entrenar(self, mensaje, etiqueta_real, max_epocas=100):
        """
        Entrena el perceptrón con un solo ejemplo.
        etiqueta_real: 1 para spam, 0 para no spam.
        """
        inputs = self.texto_a_binario(mensaje)
        for _ in range(max_epocas):
            prediccion = self.predecir(mensaje)
            error = etiqueta_real - prediccion

            # Si no hay error, terminar
            if error == 0:
                break

            # Ajustar pesos y sesgo
            for i in range(self.input_size):
                self.weights[i] += self.learning_rate * error * inputs[i]
            self.bias += self.learning_rate * error

    def entrenar_lote(self, ejemplos, etiquetas, max_epocas=100):
        """Entrena con múltiples ejemplos"""
        for _ in range(max_epocas):
            errores = 0
            for mensaje, etiqueta in zip(ejemplos, etiquetas):
                inputs = self.texto_a_binario(mensaje)
                prediccion = self.predecir(mensaje)
                error = etiqueta - prediccion

                if error != 0:
                    errores += 1
                    for i in range(self.input_size):
                        self.weights[i] += self.learning_rate * error * inputs[i]
                    self.bias += self.learning_rate * error

            # Si no hay errores, terminar
            if errores == 0:
                break


def modo_prueba(perceptron):
    """Función para probar el perceptrón en tiempo real"""
    print("\n--- Modo Prueba ---")
    print("Ingrese mensajes para clasificar (deje vacío para salir):")

    while True:
        mensaje = input("\nMensaje a clasificar: ").strip()
        if not mensaje:
            break

        prediccion = perceptron.predecir(mensaje)
        clasificacion = "Se avecina mal tiempo" if prediccion == 1 else "Relajate habra buen tiempo"
        print(f"Resultado: {clasificacion}")


def main():
    # Datos de entrenamiento predefinidos: estado del clima en la tarde en Cusco
    mensajes_entrenamiento = [
        "Está soleando fuerte, pero el viento está helado, 18°C.",  # buen tiempo
        "Cielo gris, mucho viento y se siente húmedo, 14°C.",  # mal tiempo
        "Hay sol, pero unas nubes raras, aire fresco, 17°C.",  # buen tiempo
        "Está lloviendo suave, todo nublado, frío, 13°C.",  # mal tiempo
        "Soleado, cielo limpio, brisita rica, 19°C.",  # buen tiempo
        "Nublado, viento fuerte, como que viene lluvia, 15°C.",  # mal tiempo
        "Solazo, ni una nube, aire seco, 18°C.",  # buen tiempo
        "Cielo oscuro, humedad heavy, viento raro, 14°C.",  # mal tiempo
        "Sol con nubecitas, se siente bien, 16°C.",  # buen tiempo
        "Llovizna y viento fuerte, está helando, 12°C.",  # mal tiempo
        "Tarde clarita, algo de nubes, brisa suave, 17°C.",  # buen tiempo
        "Lluvia fuerte, cielo negro, viento loco, 13°C.",  # mal tiempo
        "Está soleado, cielo azul, viento leve, 19°C.",  # buen tiempo
        "Nubes pesadas, viento que jala, húmedo, 15°C.",  # mal tiempo
        "Sol rico, pocas nubes, aire fresquito, 16°C.",  # buen tiempo
        "Chubascos, viento fuerte, frío, 14°C.",  # mal tiempo
        "Tarde soleada, cielo despejado, brisa, 18°C.",  # buen tiempo
        "Nublado, humedad alta, viento feo, 13°C.",  # mal tiempo
        "Sol con nubes ligeras, se siente fresco, 17°C.",  # buen tiempo
        "Tormenta con truenos, viento y lluvia, 12°C.",  # mal tiempo
    ]
    etiquetas_entrenamiento = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # Configuración inicial
    max_length = max(len(m) for m in mensajes_entrenamiento) + 5  # Longitud basada en los datos de entrenamiento
    perceptron = PerceptronSpam(max_length=max_length)

    # Entrenar con los datos predefinidos
    print("Entrenando con datos predefinidos...")
    perceptron.entrenar_lote(mensajes_entrenamiento, etiquetas_entrenamiento)
    print(f"Modelo entrenado con {len(mensajes_entrenamiento)} ejemplos")

    # Menú principal
    while True:
        print("\n--- Menú Principal ---")
        print("1. Probar modelo con nuevos mensajes")
        print("2. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == '1':
            modo_prueba(perceptron)
        elif opcion == '2':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Por favor intente nuevamente.")


if __name__ == "__main__":
    main()