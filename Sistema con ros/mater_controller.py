#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial  # Librería para comunicación serial
import time
from geometry_msgs.msg import Twist     # Para suscribirse a /cmd_vel
from std_msgs.msg import Bool           # Para suscribirse a topics de control

# --- Configuración de Conexión ---
# Asegúrate de que tu Pi tenga habilitado el UART en /dev/ttyAMA0
# y que tengas permisos (sudo chmod 666 /dev/ttyAMA0)
SERIAL_PORT = '/dev/ttyAMA0' 
SERIAL_BAUD = 115200 # Debe coincidir con el ESP32 

# --- Geometría del Robot (de tu main.ino) ---
# Necesaria para convertir Twist (v, w) en velocidades de rueda (wL, wR)
WHEEL_RADIUS = 0.0335  # (R) Radio de la llanta [cite: 68]
WHEEL_BASE = 0.187     # (d) Distancia entre llantas [cite: 68]

class MasterControllerNode(Node):
    """
    Este nodo Maestro escucha los topics de ROS 2 (como /cmd_vel)
    y los traduce a comandos seriales para el Esclavo (ESP32).
    """
    def __init__(self):
        super().__init__('master_controller_node')
        self.get_logger().info('Nodo Maestro (Puente ROS-Serial) iniciado.')

        # --- Iniciar conexión serial ---
        try:
            self.esp32_serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1.0)
            time.sleep(2) # Esperar a que el ESP32 se reinicie
            self.get_logger().info(f'Conectado a ESP32 en {SERIAL_PORT} a {SERIAL_BAUD} baudios.')
        except Exception as e:
            self.get_logger().fatal(f'No se pudo conectar al ESP32: {e}')
            rclpy.shutdown() # Apaga el nodo si no hay conexión
            return

        # --- Suscriptores de ROS 2 ---
        
        # 1. Suscriptor a /cmd_vel (el comando de la Red Neuronal)
        # La Red Neuronal publicará un Twist msg (linear.x, angular.z)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback, # Función a llamar
            10)

        # 2. Suscriptor para la Succión
        self.suction_sub = self.create_subscription(
            Bool,
            '/control/suction', # Un topic simple para encender/apagar
            self.suction_callback,
            10)
        
        # 3. Suscriptor para los Cepillos
        self.brushes_sub = self.create_subscription(
            Bool,
            '/control/brushes',
            self.brushes_callback,
            10)

        # 4. Suscriptor para la Aspersión
        self.aspersion_sub = self.create_subscription(
            Bool,
            '/control/aspersion',
            self.aspersion_callback,
            10)
            
        self.get_logger().info('Suscriptores creados. Esperando comandos...')

    def send_command_to_slave(self, command_str):
        """
        Función helper para enviar el comando formateado al ESP32.
        """
        try:
            command = f"<{command_str}>"
            self.esp32_serial.write(command.encode('utf-8'))
            self.get_logger().debug(f'Enviado: {command}')
        except Exception as e:
            self.get_logger().warn(f'Error al enviar comando serial: {e}')

    # --- Funciones de Callback ---

    def cmd_vel_callback(self, msg: Twist):
        """
        Recibe un mensaje Twist (v, w) y lo convierte a wL y wR.
        """
        # v = Velocidad lineal (msg.linear.x)
        # w = Velocidad angular (msg.angular.z)
        v = msg.linear.x
        w = msg.angular.z

        # --- Cinemática Inversa (Robot Diferencial) ---
        # wL = (v / R) - (w * d / (2 * R))
        # wR = (v / R) + (w * d / (2 * R))
        
        wL = (v / WHEEL_RADIUS) - (w * WHEEL_BASE / (2 * WHEEL_RADIUS))
        wR = (v / WHEEL_RADIUS) + (w * WHEEL_BASE / (2 * WHEEL_RADIUS))
        
        # Formatea el comando de motor
        cmd = f"M,{wL:.3f},{wR:.3f}"
        self.send_command_to_slave(cmd)

    def suction_callback(self, msg: Bool):
        """ Recibe un booleano para la succión """
        state = 1 if msg.data else 0
        cmd = f"S,{state}"
        self.send_command_to_slave(cmd)

    def brushes_callback(self, msg: Bool):
        """ Recibe un booleano para los cepillos """
        state = 1 if msg.data else 0
        cmd = f"C,{state}"
        self.send_command_to_slave(cmd)

    def aspersion_callback(self, msg: Bool):
        """ Recibe un booleano para la aspersión """
        state = 1 if msg.data else 0
        cmd = f"A,{state}"
        self.send_command_to_slave(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MasterControllerNode()
    rclpy.spin(node)
    
    # Al cerrar (Ctrl+C), detener todo
    node.get_logger().info('Apagando... deteniendo motores.')
    node.send_command_to_slave("M,0.0,0.0") # Detiene motores
    node.send_command_to_slave("S,0") # Detiene succión
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()