#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import tflite_runtime.interpreter as tflite # Importa el runtime ligero

# --- Parámetros de Configuración ---
NUM_LIDAR_SAMPLES = 60 # ¡DEBE COINCIDIR CON LOS OTROS SCRIPTS!
MAX_LIDAR_RANGE = 5.0  # ¡DEBE COINCIDIR!
TFLITE_MODEL_PATH = 'navigation_model.tflite' # Ruta al modelo

# Escala de los comandos. 'tanh' da de -1 a 1.
# Ajusta estos valores según la velocidad máxima de tu robot.
MAX_LINEAR_SPEED = 0.5  # metros/segundo
MAX_ANGULAR_SPEED = 1.0 # radianes/segundo
# ------------------------------------

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.get_logger().info("Iniciando Nodo de Navegación Neuronal")

        # --- Cargar el modelo TFLite ---
        try:
            self.interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.get_logger().info("Modelo TFLite cargado exitosamente.")
        except Exception as e:
            self.get_logger().fatal(f"Error al cargar el modelo TFLite: {e}")
            rclpy.shutdown()
            return

        # --- Suscriptor al Lidar ---
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)

        # --- Publicador de Comandos ---
        # Este nodo PUBLICA en /cmd_vel, que es el topic que
        # escucha tu 'master_controller.py'
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info("Nodo listo. ¡Navegación autónoma iniciada!")

    def preprocess_lidar(self, msg: LaserScan):
        """
        Pre-procesa el escaneo del Lidar.
        DEBE SER IDÉNTICO AL PRE-PROCESADO DEL LOGGER.
        """
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = MAX_LIDAR_RANGE
        ranges[np.isnan(ranges)] = MAX_LIDAR_RANGE
        
        indices = np.linspace(0, len(ranges) - 1, NUM_LIDAR_SAMPLES, dtype=int)
        sampled_ranges = ranges[indices]
        
        sampled_ranges = np.clip(sampled_ranges, 0.0, MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE
        
        # El modelo espera una entrada con forma (1, NUM_LIDAR_SAMPLES)
        return np.expand_dims(sampled_ranges, axis=0).astype(np.float32)

    def lidar_callback(self, msg: LaserScan):
        """
        Función principal: Recibe Lidar, predice y publica comando.
        """
        # 1. Pre-procesar los datos del Lidar
        input_data = self.preprocess_lidar(msg)

        # 2. Ejecutar Inferencia (Predicción)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # 3. Obtener la predicción
        # La salida es un array (ej. [[0.8, -0.2]])
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        linear_pred = prediction[0]
        angular_pred = prediction[1]

        # 4. Crear el mensaje Twist
        twist_msg = Twist()
        # Escalamos la salida de 'tanh' (-1 a 1) a nuestras velocidades máximas
        twist_msg.linear.x = float(linear_pred * MAX_LINEAR_SPEED)
        twist_msg.angular.z = float(angular_pred * MAX_ANGULAR_SPEED)

        # 5. Publicar el comando
        self.cmd_vel_pub.publish(twist_msg)
        self.get_logger().info(f"Publicando: Linear={twist_msg.linear.x:.2f}, Angular={twist_msg.angular.z:.2f}", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    nav_node = NavigationNode()
    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Al cerrar, publica un comando de detención
        stop_msg = Twist()
        nav_node.cmd_vel_pub.publish(stop_msg)
        nav_node.get_logger().info("Deteniendo el robot.")
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()