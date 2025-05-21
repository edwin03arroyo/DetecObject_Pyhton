import cv2
import numpy as np

# Cargar nombres de clases desde coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Cargar red neuronal (pesos y configuración)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Obtener nombres de capas de salida
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Iniciar captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Convertir imagen a blob para el modelo
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Procesar resultados
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                cx, cy, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        # Compatibilidad con distintas versiones de OpenCV
        if isinstance(indexes, tuple):
            indexes = indexes[0]

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detector de Objetos - YOLOv3", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
