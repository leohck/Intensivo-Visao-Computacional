import cv2
from datetime import datetime

DATE_FORMAT = "%d/%m/%Y"
TIME_FORMAT = "%H:%M:%S"
SHORT_DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"


def pegar_data_hora_atual():
    """
    Retorna a data e hora atual
    :return data, hora, data_hora
    :rtype list
    """
    atual = datetime.now()
    data = atual.date().strftime(DATE_FORMAT)
    hora = atual.time().strftime(TIME_FORMAT)
    data_hora = atual.strftime(SHORT_DATETIME_FORMAT)
    return data, hora, data_hora


def tirar_foto(frame, dimensoes, file_name):
    """
    Lida com o registro de fotos
    :param frame numpy.ndarray = O frame de onde a foto deve ser registrada
    :param dimensoes numpy.ndarray = As dimensoes (x,y,w,h) para criar um novo frame a apartir das dimensoes informadas
    :param file_name str = O nome do arquivo para salvar a imagem
    :return None
    :rtype None
    """
    (x, y, w, h) = dimensoes[0], dimensoes[1], dimensoes[2], dimensoes[3]
    file = f"imagens/{file_name}.png"
    imagem = frame[y:y + h, x:x + w]
    cv2.namedWindow("foto", cv2.WINDOW_AUTOSIZE)
    cv2.imshow('foto', imagem)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('e'):
        cv2.destroyWindow("foto")
    elif key == ord('s'):
        cv2.imwrite(file, imagem)
        cv2.destroyWindow("foto")
        print(f'foto salva -> {file_name}')


def main():
    car_classificador = cv2.CascadeClassifier('cascades/cars2.xml')
    camera = cv2.VideoCapture('videos/rodovia.mp4')
    reproduzir = False  # se True o video ja inicia sendo reproduzido
    detectar_carros = False  # se True o video ja inicia detectando carros
    mostrar_comandos = True  # se False nao aparece o menu com os comandos
    habilitar_comandos = True  # se False os comandos não funcionam apenas o ESC para encerrar o programa
    while True:
        retval, frame = camera.read()
        frame = cv2.resize(frame, None, fx=.5, fy=.5)
        if mostrar_comandos:
            width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame = cv2.rectangle(frame, (0, 0), (int(width), int(height / 15)), (255, 255, 255), -1)
            comandos = "s - iniciar | p - pausar | e - encerrar"
            comandos2 = "d-detectar/pausar | f-foto (e-cancelar|s-salvar)"
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, comandos, (10, int(height / 40)), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, comandos2, (10, int(height / 18)), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        if habilitar_comandos:
            if not reproduzir:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    print('tecla s -> reprodução do video iniciada')
                    reproduzir = True
                elif key == ord('e'):
                    print('tecla e -> reprodução do video encerrada')
                    break
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    print('tecla p -> reprodução do video pausada')
                    reproduzir = False

                if key == ord('d'):
                    if detectar_carros:
                        print('tecla d -> detecção de veiculos encerrada')
                        detectar_carros = False
                    else:
                        print('tecla d -> detecção de veiculos iniciada')
                        detectar_carros = True

                if key == ord('e'):
                    print('tecla e -> reprodução do video encerrada')
                    break
        else:
            key = cv2.waitKey(1)
            if key == 27:
                break

        if detectar_carros:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            carros = car_classificador.detectMultiScale(gray, 1.4, 2)
            for carro in carros:
                if carro.all():
                    (x, y, w, h) = carro[0], carro[1], carro[2], carro[3]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    # A função pegar_data_hora_atual retorna 3 valores = data = [0] / hora=[1] / data e hora=[2]
                    horario = pegar_data_hora_atual()
                    # No putText estou acessando o horario com o index[2], portanto ele retorna 'data e hora'
                    cv2.putText(frame, horario[2], (int(x / 2), y), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
                    key = cv2.waitKey(1)
                    if key == ord('f'):
                        tirar_foto(frame, carro, x)

        cv2.imshow('Detector de carros', frame)

    camera.release()
    cv2.destroyAllWindows()


main()
