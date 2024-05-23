# Importar as bibliotecas necessárias
import os  # Biblioteca para interagir com o sistema de arquivos e pastas
import cv2  # Biblioteca OpenCV para processamento de imagens e detecção de rostos
import numpy as np  # Biblioteca NumPy para manipulação de arrays numéricos
import tensorflow as tf  # Biblioteca TensorFlow para criar e treinar o modelo de rede neural

# Carregar imagens de referência
known_faces = []  # Lista para armazenar as imagens de referência
known_names = []  # Lista para armazenar os nomes correspondentes
for name in os.listdir("imagens"):  # Percorrer a pasta "imagens" e suas subpastas
    dir_path = os.path.join("imagens", name)
    if os.path.isdir(dir_path):  # Verificar se é um diretório
        for image in os.listdir(dir_path):  # Percorrer as imagens dentro da subpasta
            image_path = os.path.join(dir_path, image)
            face_image = cv2.imread(image_path)  # Carregar a imagem
            known_faces.append(face_image)  # Adicionar a imagem à lista
            known_names.append(name)  # Adicionar o nome à lista

# Pré-processar imagens de referência
known_faces_encoded = []  # Lista para armazenar as imagens pré-processadas
for face in known_faces:
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Converter a imagem para RGB
    face = cv2.resize(face, (160, 160))  # Redimensionar a imagem para 160x160 pixels
    known_faces_encoded.append(face)  # Adicionar a imagem pré-processada à lista

known_faces_encoded = np.array(known_faces_encoded) / 255.0  # Converter a lista para um array NumPy e normalizar os valores de pixel

# Codificar nomes
label_encoder = tf.keras.layers.StringLookup()  # Criar uma camada StringLookup do TensorFlow
label_encoder.adapt(known_names)  # Adaptar o vocabulário da camada com os nomes das pessoas

# Criar modelo de reconhecimento facial
inputs = tf.keras.Input(shape=(160, 160, 3))  # Camada de entrada para imagens de 160x160 pixels com 3 canais (RGB)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)  # Camada convolucional com 64 filtros de 3x3 e ativação ReLU
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # Camada de pooling máximo de 2x2
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)  # Camada convolucional com 128 filtros de 3x3 e ativação ReLU
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # Camada de pooling máximo de 2x2
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)  # Camada convolucional com 256 filtros de 3x3 e ativação ReLU
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # Camada de pooling máximo de 2x2
x = tf.keras.layers.Flatten()(x)  # Achatamento das saídas das camadas convolucionais
x = tf.keras.layers.Dense(512, activation='relu')(x)  # Camada densa com 512 neurônios e ativação ReLU
x = tf.keras.layers.Dropout(0.5)(x)  # Camada de dropout com taxa de 0.5 para regularização
outputs = tf.keras.layers.Dense(len(label_encoder.get_vocabulary()), activation='softmax')(x)  # Camada de saída com número de neurônios igual ao tamanho do vocabulário codificado e ativação softmax

model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Criar o modelo de rede neural
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compilar o modelo com otimizador Adam, função de perda categorical crossentropy e métrica de acurácia
model.fit(known_faces_encoded, label_encoder(known_names), epochs=30, batch_size=32)  # Treinar o modelo com as imagens pré-processadas e os nomes codificados por 30 épocas com batch_size de 32

# Inicializar a câmera
video = cv2.VideoCapture(0)  # Inicializar a captura de vídeo da câmera padrão

while True:
    _, img = video.read()  # Ler um quadro da câmera

    # Detectar rostos na imagem
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converter o quadro para escala de cinza
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Carregar o classificador Haar Cascade para detecção de rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detectar rostos no quadro em escala de cinza

    # Reconhecer rostos
    for (x, y, w, h) in faces:
        face_image = img[y:y+h, x:x+w]  # Extrair a região do rosto do quadro
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Converter a região do rosto para RGB
        face_image = cv2.resize(face_image, (160, 160))  # Redimensionar a região do rosto para 160x160 pixels
        face_image = np.expand_dims(face_image, axis=0) / 255.0  # Adicionar uma dimensão de lote e normalizar os valores de pixel

        prediction = model.predict(face_image)  # Usar o modelo treinado para prever o nome da pessoa
        label_index = np.argmax(prediction)  # Obter o índice da classe com a maior probabilidade prevista
        confidence = prediction[0][label_index]  # Obter a confiança (probabilidade) dessa previsão

        if confidence > 0.6:  # Verificar se a confiança é maior que um limiar (0.6 neste caso)
            label = label_encoder.get_vocabulary()[label_index]  # Obter o nome correspondente ao índice previsto
        else:
            label = "Desconhecido"  # Se a confiança for baixa, definir o rótulo como "Desconhecido"

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Desenhar um retângulo em volta do rosto
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Escrever o nome previsto abaixo do retângulo

    cv2.imshow("Resultado", img)  # Exibir o quadro resultante com os rostos detectados e nomes identificados
    if cv2.waitKey(1) == 27:  # Sair do loop se a tecla 'Esc' for pressionada
        break

video.release()  # Liberar a captura de vídeo da câmera
cv2.destroyAllWindows()  # Fechar todas as janelas abertas pelo OpenCV
