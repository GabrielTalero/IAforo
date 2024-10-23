import yaml
import cv2
import time
from ultralytics import YOLO

# Cargar el modelo
model = YOLO('C:/Users/taler/iaforo/iaforo/data/best.pt')

# Definir una función para guardar los datos en YAML
def save_vehicle_data(data, filename='C:/Users/taler/iaforo/iaforo/detected_vehicles.yaml'):
    # Abrimos el archivo en modo 'append' para que no sobrescriba los datos anteriores
    with open(filename, 'a') as file:
        yaml.dump([data], file, default_flow_style=False)

# Inicializar la captura de video desde la cámara (cámara 0)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/Users/taler/iaforo/iaforo/data/video11.mp4')

# Crear la ventana antes de la detección para evitar errores
cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)  # Crear ventana ajustable

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener las dimensiones actuales de la ventana (para el ajuste automático)
    window_rect = cv2.getWindowImageRect('YOLO Detection')
    current_width = window_rect[2]
    current_height = window_rect[3]

    # Verificar si las dimensiones son válidas (mayores que cero)
    if current_width > 0 and current_height > 0:
        # Redimensionar el frame a las dimensiones actuales de la ventana
        resized_frame = cv2.resize(frame, (current_width, current_height))
    else:
        # Si las dimensiones no son válidas, usa el frame original sin redimensionar
        resized_frame = frame

    # Redimensionar el frame a las dimensiones actuales de la ventana
    #resized_frame = cv2.resize(frame, (current_width, current_height))

    # Realizar inferencia en el frame redimensionado
    results = model.track(resized_frame, tracker="bytetrack.yaml", conf=0.5)

    # Mostrar los resultados redimensionados en pantalla
    cv2.imshow('YOLO Detection', results[0].plot())


    # Procesar cada resultado de la detección
    for box in results[0].boxes:
        print("Atributos del box:", box)
        # Obtener la hora actual en formato de timestamp
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        # Extraer datos relevantes
        data = {
            'id': int(box.id) if box.id is not None else None,  # Guardar la ID de la box si existe
            'class': results[0].names[int(box.cls)],
            'time': current_time,  # Tiempo en que se detectó el vehículo
            'confidence': float(box.conf),
            'coordinates': box.xywh.tolist()  # Guardamos coordenadas del bounding box
        }

        # Guardar el dato en el archivo YAML
        save_vehicle_data(data)

    print("Datos guardados en tiempo real")

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
