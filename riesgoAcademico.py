import random
import math


class PerceptronRiesgoAcademico:
    def __init__(self):
        """
        Inicializa el perceptrón con las siguientes entradas:
        - llega_tarde: 1 bit (0 o 1)
        - promedio_tareas: 5 bits (0-20)
        - promedio_examenes: 5 bits (0-20)
        - porcentaje_asistencia: 7 bits (0-100)
        - es_sociable: 1 bit (0 o 1)
        Total: 1 + 5 + 5 + 7 + 1 = 19 entradas
        """
        self.input_size = 19
        self.weights = [random.uniform(-1, 1) for _ in range(self.input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.05

    def convertir_a_binario(self, valor, bits):
        """Convierte un valor numérico a su representación binaria con la cantidad de bits especificada"""
        # Primero redondeamos el valor a entero
        valor_int = int(round(valor))
        binary = []
        for i in range(bits):
            binary.append((valor_int >> (bits - 1 - i)) & 1)
        return binary

    def preparar_entradas(self, llega_tarde, promedio_tareas, promedio_examenes, porcentaje_asistencia, es_sociable):
        """
        Prepara las entradas para el perceptrón convirtiendo todos los valores a binario
        """
        inputs = []

        # Llega tarde (1 bit)
        inputs.append(llega_tarde)

        # Promedio de tareas (5 bits, rango 0-20)
        inputs.extend(self.convertir_a_binario(min(max(promedio_tareas, 0), 20), 5))

        # Promedio de exámenes (5 bits, rango 0-20)
        inputs.extend(self.convertir_a_binario(min(max(promedio_examenes, 0), 20), 5))

        # Porcentaje de asistencia (7 bits, rango 0-100)
        inputs.extend(self.convertir_a_binario(min(max(porcentaje_asistencia, 0), 100), 7))

        # Es sociable (1 bit)
        inputs.append(es_sociable)

        return inputs

    def activacion(self, x):
        """Función de activación escalón (step function)"""
        return 1 if x >= 0 else 0

    def predecir(self, llega_tarde, promedio_tareas, promedio_examenes, porcentaje_asistencia, es_sociable):
        """Predice si el alumno está en alto riesgo (1) o bajo riesgo (0)"""
        inputs = self.preparar_entradas(llega_tarde, promedio_tareas, promedio_examenes, porcentaje_asistencia,
                                        es_sociable)

        # Calcular suma ponderada
        z = self.bias
        for i in range(self.input_size):
            z += self.weights[i] * inputs[i]

        # Aplicar función de activación
        return self.activacion(z)

    def entrenar(self, datos_entrenamiento, etiquetas, max_epocas=1000):
        """Entrena el perceptrón con los datos de entrenamiento"""
        for _ in range(max_epocas):
            errores = 0
            for datos, etiqueta in zip(datos_entrenamiento, etiquetas):
                llega_tarde, promedio_tareas, promedio_examenes, porcentaje_asistencia, es_sociable = datos
                inputs = self.preparar_entradas(llega_tarde, promedio_tareas, promedio_examenes, porcentaje_asistencia,
                                                es_sociable)
                z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
                prediccion = self.activacion(z)
                error = etiqueta - prediccion

                if error != 0:
                    errores += 1
                    for i in range(self.input_size):
                        self.weights[i] += self.learning_rate * error * inputs[i]
                    self.bias += self.learning_rate * error

            # Si no hay errores, terminar antes
            if errores == 0:
                break


# Datos de entrenamiento predefinidos
# Cada tupla contiene: (llega_tarde, promedio_tareas, promedio_examenes, porcentaje_asistencia, es_sociable)
datos_entrenamiento = [
    (1, 8.5, 7.3, 65.2, 0),  # Alto riesgo
    (0, 18.2, 17.1, 95.0, 1),  # Bajo riesgo
    (1, 10.0, 9.5, 70.1, 0),  # Alto riesgo
    (0, 15.3, 16.2, 90.5, 1),  # Bajo riesgo
    (1, 6.7, 5.8, 60.0, 0),  # Alto riesgo
    (0, 19.0, 18.5, 98.3, 1),  # Bajo riesgo
    (1, 9.2, 8.7, 68.4, 0),  # Alto riesgo
    (0, 16.1, 15.8, 92.7, 1),  # Bajo riesgo
    (1, 7.9, 6.5, 63.8, 0),  # Alto riesgo
    (0, 17.5, 16.9, 96.2, 1)  # Bajo riesgo
]

etiquetas_entrenamiento = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def main():
    # Crear y entrenar el perceptrón
    perceptron = PerceptronRiesgoAcademico()
    perceptron.entrenar(datos_entrenamiento, etiquetas_entrenamiento)
    print("Perceptrón entrenado con datos predefinidos")

    # Modo de prueba interactivo
    print("\n--- Modo de Prueba ---")
    print("Ingrese los datos del alumno para predecir su riesgo (deje vacío para salir)")

    while True:
        print("\nDatos del alumno:")
        try:
            llega_tarde = input("¿Llega tarde frecuentemente? (1=Sí, 0=No): ").strip()
            if not llega_tarde:
                break
            llega_tarde = int(llega_tarde)

            promedio_tareas = float(input("Promedio de notas de tareas (0-20): "))
            promedio_examenes = float(input("Promedio de notas de exámenes (0-20): "))
            porcentaje_asistencia = float(input("Porcentaje de asistencia (0-100): "))
            es_sociable = int(input("¿Es sociable? (1=Sí, 0=No): "))

            # Validar entradas
            if (llega_tarde not in [0, 1] or
                    not (0 <= promedio_tareas <= 20) or
                    not (0 <= promedio_examenes <= 20) or
                    not (0 <= porcentaje_asistencia <= 100) or
                    es_sociable not in [0, 1]):
                print("Error: Datos inválidos. Por favor ingrese valores correctos.")
                continue

            # Predecir riesgo
            riesgo = perceptron.predecir(llega_tarde, promedio_tareas, promedio_examenes, porcentaje_asistencia,
                                         es_sociable)
            resultado = "ALTO RIESGO" if riesgo == 1 else "BAJO RIESGO"
            print(f"\nResultado: El alumno está en {resultado}")

        except ValueError:
            print("Error: Por favor ingrese valores numéricos válidos.")


if __name__ == "__main__":
    main()