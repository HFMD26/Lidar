#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import time

# --- Parámetros de Configuración ---
NUM_LIDAR_SAMPLES = 60 # Reducimos los 360+ puntos del Lidar a solo 60
MAX_LIDAR_RANGE = 5.0  # Distancia máxima (en metros) que nos importa
DATASET_FILE_NAME = f'dataset_{int(time.time())}.npz' # Nombre del archivo de salida
# ------------------------------------

class DataLoggerNode(Node):
    def __init__(self):
        super().__init__('data_logger_node')
        self.get_logger().info(f"Iniciando Data Logger. Guardando en: {DATASET_FILE_NAME}")
        self.get_logger().warn("POR FAVOR, PUBLIQUE LOS COMANDOS MANUALES EN /cmd_vel_human")

        # Suscriptor al Lidar
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)

        # Suscriptor al comando manual (joystick, teclado, etc.)
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel_human', # Asegúrate de que tu teleop publique aquí
            self.cmd_vel_callback,
            10)

        self.lidar_data = None
        self.last_cmd = (0.0, 0.0) # (linear.x, angular.z)
        self.all_lidar_samples = []
        self.all_commands = []

        # Timer para guardar datos periódicamente
        self.save_timer = self.create_timer(0.1, self.save_data_point) # Guarda datos 10 veces por segundo
        self.get_logger().info("Nodo listo. Conduce el robot...")

    def preprocess_lidar(self, msg: LaserScan):
        """
        Pre-procesa el escaneo del Lidar:
        1. Limpia 'inf' y 'nan'.
        2. Reduce el número de muestras (sampling).
        """
        ranges = np.array(msg.ranges)
        
        # 1. Limpia 'inf' (infinito) y 'nan' (no es un número)
        ranges[np.isinf(ranges)] = MAX_LIDAR_RANGE
        ranges[np.isnan(ranges)] = MAX_LIDAR_RANGE
        
        # 2. Muestreo (Sampling): Toma N muestras equidistantes
        indices = np.linspace(0, len(ranges) - 1, NUM_LIDAR_SAMPLES, dtype=int)
        sampled_ranges = ranges[indices]
        
        # 3. Normaliza los datos (opcional pero recomendado)
        sampled_ranges = np.clip(sampled_ranges, 0.0, MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE
        
        return sampled_ranges

    def lidar_callback(self, msg: LaserScan):
        """ Almacena el último dato pre-procesado del Lidar """
        self.lidar_data = self.preprocess_lidar(msg)

    def cmd_vel_callback(self, msg: Twist):
        """ Almacena el último comando manual """
        self.last_cmd = (msg.linear.x, msg.angular.z)

    def save_data_point(self):
        """ Guarda el par (Lidar, Comando) actual si el robot se está moviendo """
        if self.lidar_data is None:
            return # Aún no recibimos Lidar

        # Solo guarda si el comando no es cero (para evitar datos sesgados)
        if self.last_cmd[0] != 0.0 or self.last_cmd[1] != 0.0:
            self.all_lidar_samples.append(self.lidar_data)
            self.all_commands.append(self.last_cmd)
            self.get_logger().info(f"Punto de dato guardado. Total: {len(self.all_commands)}", throttle_duration_sec=1.0)

    def save_to_file(self):
        """ Guarda el dataset completo en un archivo .npz """
        if not self.all_commands:
            self.get_logger().warn("No se guardaron datos, el dataset está vacío.")
            return

        self.get_logger().info(f"Guardando {len(self.all_commands)} puntos en {DATASET_FILE_NAME}...")
        np.savez_compressed(
            DATASET_FILE_NAME,
            x_lidar=np.array(self.all_lidar_samples),
            y_cmd=np.array(self.all_commands)
        )
        self.get_logger().info("¡Dataset guardado con éxito!")

def main(args=None):
    rclpy.init(args=args)
    logger_node = DataLoggerNode()
    try:
        rclpy.spin(logger_node)
    except KeyboardInterrupt:
        logger_node.get_logger().info("Cerrando... guardando archivo de datos.")
        logger_node.save_to_file()
    finally:
        logger_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()