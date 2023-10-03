import cv2 as cv
import numpy as np
 # Зчитуємо фотку
img = cv.imread("images/text_2/2.jpg")
 # Створюємо масив з сірим відтінком
grey = []
for RGB_Pixel in img:
    list_num_gray = []
    for grey_pixel in RGB_Pixel:
        list_num_gray.append(sum(grey_pixel) // 3)
    grey.append(list_num_gray)  
 
 # Створюємо об'єкт з сірими відтінками
grey_img = np.array(grey, dtype=np.uint8)
img_grey = cv.imwrite("images/text_2/grey_image.jpg", grey_img)


# Адаптивна бінаризація. метод Крістіана
def adaptive_binarization(image, window_size, c):
    # Отримуємо розмір зображення
    height, width = image.shape

    # Створюємо вихідне зображення для результату бінаризації
    binarized_image = np.zeros((height, width), dtype=np.uint8)

    # Проходимо по кожному пікселю в зображенні
    for y in range(height):
        for x in range(width):
            # Обчислюємо локальний поріг бінаризації за методом Крістіана
            sum_pixels = 0
            count_pixels = 0
            for i in range(-window_size // 2, window_size // 2 + 1):
                for j in range(-window_size // 2, window_size // 2 + 1):
                    if 0 <= y + i < height and 0 <= x + j < width:
                        sum_pixels += image[y + i, x + j]
                        count_pixels += 1

            local_threshold = (sum_pixels / count_pixels) - c

            # Виконуємо бінаризацію
            if image[y, x] > local_threshold:
                binarized_image[y, x] = 255
            else:
                binarized_image[y, x] = 0

    return binarized_image

# Зчитуємо чорно-біле зображення (відтінки сірого)
input_image = cv.imread('images/text_2/grey_image.jpg', cv.IMREAD_GRAYSCALE)

# Викликаємо адаптивну бінаризацію зі своїми параметрами
window_size = 15
c = 0.5
result_image = adaptive_binarization(input_image, window_size, c)

# Інвертуємо кольори (чорний на білому фоні до білого на чорному фоні)
inverted_binary_image = cv.bitwise_not(result_image)

# Загружаємо результат(и)
cv.imwrite("images/text_2/binar_image.jpg", result_image)
cv.imwrite("images/text_2/inverted_binary_image.jpg", inverted_binary_image)

# Знаходимо контури об'єкта на бінарному зображенні
contours, _ = cv.findContours(inverted_binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Створюємо порожню маску з такими ж розмірами, як і вхідне зображення
mask = np.zeros_like(inverted_binary_image, dtype=np.uint8)

# Малюємо контури об'єкта на масці
cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)

# Вирізаємо об'єкт з оригінального зображення за допомогою маски
object_cut = cv.bitwise_and(img, img, mask=mask)

# Зберігаємо вирізаний об'єкт
cv.imwrite("images/text_2/image_cut.jpg", object_cut)



