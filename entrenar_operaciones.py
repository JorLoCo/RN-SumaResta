import tensorflow as tf
import numpy as np
import os
import sys
import subprocess

# Función auxiliar para crear y guardar modelos
def entrenar_y_guardar(operacion, nombre_carpeta):
    print(f"\n--- 🧠 Entrenando cerebro para: {operacion.upper()} ---")
    
    # 1. Datos Sintéticos (10,000 muestras)
    # Generamos pares de números entre -1000 y 1000
    X = np.random.randint(-1000, 1000, (10000, 2)).astype(float)
    
    if operacion == 'suma':
        y = np.sum(X, axis=1) # Suma las columnas
    else:
        y = X[:, 0] - X[:, 1] # Resta: Columna 0 - Columna 1

    # 2. Modelo (2 entradas -> 4 ocultas -> 1 salida)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_shape=[2], activation='linear'), 
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    
    # Entrenamos
    model.fit(X, y, epochs=50, verbose=0)
    print("¡Entrenamiento finalizado!")
    
    # Prueba rápida
    test_data = np.array([[10, 5]])
    pred = model.predict(test_data)
    print(f"Prueba (10, 5): Resultado IA = {pred[0][0]:.2f}")

    # 3. Guardar y Convertir
    keras_file = f'{operacion}.h5'
    model.save(keras_file)
    
    output_path = f'public/{nombre_carpeta}' # Guardamos directo en estructura para copiar fácil
    os.makedirs(output_path, exist_ok=True)
    
    print("Convirtiendo a TensorFlow.js...")
    command = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=keras", keras_file, output_path
    ]
    subprocess.run(command, check=True)
    print(f"✅ Modelo guardado en {output_path}")

# Ejecutar para ambas operaciones
if __name__ == "__main__":
    entrenar_y_guardar('suma', 'modelo_suma')
    entrenar_y_guardar('resta', 'modelo_resta')