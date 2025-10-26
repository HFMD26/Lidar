import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- Parámetros ---
DATASET_FILE = 'dataset_1729443600.npz' # <--- CAMBIA ESTO por el nombre de tu archivo
NUM_LIDAR_SAMPLES = 60 # Debe coincidir con el script del Logger
MODEL_NAME = 'navigation_model'
# ------------------

def load_data(file_name):
    """ Carga los datos del archivo .npz """
    if not os.path.exists(file_name):
        print(f"Error: No se encontró el archivo {file_name}")
        return None, None
        
    print(f"Cargando dataset desde {file_name}...")
    data = np.load(file_name)
    x = data['x_lidar']
    y = data['y_cmd']
    print(f"Datos cargados: {len(x)} muestras.")
    return x, y

def build_model():
    """ Construye la arquitectura de la Red Neuronal """
    model = Sequential([
        Input(shape=(NUM_LIDAR_SAMPLES,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        # Capa de salida: 2 neuronas (linear.x, angular.z)
        # Usamos 'tanh' para que la salida esté entre -1 y 1
        Dense(2, activation='tanh') 
    ])
    
    # Usamos 'mean_squared_error' porque es un problema de regresión
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def plot_history(history):
    """ Grafica la pérdida del entrenamiento """
    plt.figure()
    plt.plot(history.history['loss'], label='Pérdida (loss) de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida (loss) de validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Historial de Entrenamiento')
    plt.show()

def main():
    # 1. Cargar datos
    x_data, y_data = load_data(DATASET_FILE)
    if x_data is None:
        return

    # 2. Dividir en entrenamiento y validación
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    # 3. Construir y entrenar el modelo
    model = build_model()
    
    print("\nIniciando entrenamiento...")
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    )
    print("Entrenamiento finalizado.")
    
    # 4. Guardar el modelo de Keras (completo)
    model.save(f'{MODEL_NAME}.h5')
    print(f"Modelo Keras guardado como {MODEL_NAME}.h5")

    # 5. --- Convertir y Guardar el Modelo TensorFlow Lite ---
    # Este es el modelo que usaremos en la Raspberry Pi
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Optimiza el modelo
    tflite_model = converter.convert()

    with open(f'{MODEL_NAME}.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"¡Éxito! Modelo convertido y guardado como {MODEL_NAME}.tflite")
    
    # 6. Graficar resultados
    plot_history(history)

if __name__ == '__main__':
    main()