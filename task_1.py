"""
1 задание по варианту
"""

import cv2


def preprocess_image(image_path="variant-2.png", result_path="blurred_image.jpg"):
    """
    Предобработка изображения
    :param image_path:  Путь к файлу
    :param result_path: Путь для сохранения
    """
    # загрузить изображение
    image = cv2.imread(image_path)

    # если image == None, значит такого файла нет
    if image is None:
        print('Изображение не было открыто.')
        exit()

    # задать размер ядра фильтрации
    ksize = (15, 15)

    # задать стандартное отклонение
    sigmaX = 0

    # применить размытие по Гауссу
    blurred = cv2.GaussianBlur(image, ksize, sigmaX)

    # сохранить изображение в файл
    cv2.imwrite(result_path, blurred)

    # показать изображение
    cv2.imshow("Gaussian Blur", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
