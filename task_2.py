"""
2 задание по варианту
"""

import cv2
import numpy as np


def check_photo(template_path="ref-point.jpg", photo_path="test.png"):
    """
    Поиск метки по примеру
    :param template_path: Что ищем
    :param photo_path:    Где ищем
    """
    # загрузить изображение метки
    template = cv2.imread(template_path)

    if template is None:
        print('Не загрузился пример')
        return

    # преобразовать изображение метки в оттенки серого
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # применить пороговую обработку к изображению метки
    ret, thresh = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)

    # найти контуры на изображении метки
    template_contours, template_hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # получить контур шаблона
    template_contour = template_contours[0]

    # загрузить изображение для поиска метки
    image = cv2.imread(photo_path)

    if image is None:
        print('Не загрузилось фото')
        return

    # преобразовать изображение для поиска в оттенки серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # выполнить шаблонное сопоставление
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # получить координаты метки
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    center = (max_loc[0] + template.shape[1] / 2, max_loc[1] + template.shape[0] / 2)

    # найти контуры на изображении
    contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # выполнить дополнительную проверку на соответствие контура шаблону
    match = False
    for cnt in contours:
        # вычислить коэффициент сходства между контуром и шаблоном
        similarity = cv2.matchShapes(cnt, template_contour, cv2.CONTOURS_MATCH_I3, 0)

        # если коэффициент сходства больше заданного значения, то считаем, что контур соответствует метке
        if similarity > 0.9:
            print(similarity)
            match = True
            break

    if match is True:
        # вывести координаты центра метки
        print("Координаты центра метки: ({}, {})".format(center[0], center[1]))

        # сохранить координаты центра метки в файл
        np.savetxt("center_coords.txt", np.array(center), fmt='%.3f')

        # нарисовать прямоугольник на изображении
        cv2.rectangle(image, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)

        # сохранить изображение с нарисованным прямоугольником
        cv2.imwrite("labeled_other_image.png", image)
    else:
        print('Не найдено')


# Запуск функции
check_photo()
