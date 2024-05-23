Desenvolvi esse projeto de reconhecimento facial em tempo real, utilizando tecnologias de ponta em Inteligência Artificial e Machine Learning. Aqui está um resumo do que ele faz:

Construi um sistema capaz de carregar imagens de referência de uma pasta, pré-processá-las e codificar os nomes das pessoas. Em seguida, criei e treinei um modelo de rede neural convolucional (CNN) usando o TensorFlow para realizar o reconhecimento facial.

O sistema que eu desenvolvi inicializa a câmera e entra em um loop infinito para capturar quadros em tempo real. Em cada quadro, ele detecta rostos usando o classificador Haar Cascade do OpenCV. Para cada rosto detectado, o sistema extrai a região do rosto, pré-processa e passa pelo modelo CNN treinado para prever o nome da pessoa.

O projeto desenha um retângulo em volta do rosto e exibe o nome previsto abaixo do retângulo, apresentando o quadro resultante com os rostos detectados e nomes identificados.

As principais tecnologias que eu utilizei nesse projeto são:

Inteligência Artificial: Reconhecimento facial usando redes neurais convolucionais (CNN).

Machine Learning: Treinamento de um modelo CNN usando TensorFlow.

Processamento de Imagens: OpenCV para detecção de rostos e manipulação de imagens.

Bibliotecas: TensorFlow, OpenCV, NumPy.
