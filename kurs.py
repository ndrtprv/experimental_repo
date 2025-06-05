# %%
import numpy as np # бібліотека для роботи з масивами
import cv2 # бібліотека для роботи з комп'ютерним зором
from matplotlib import pyplot as plt # бібліотека для створення та відображення графіків

image_filenames = ['test/1.bmp', 'test/2.bmp', 'test/3.bmp', 'test/4.bmp'] # зображення класів розпізнавання
big_image = "test/intel.jpg" # велике зображення
class_names = ['A', 'B', 'C', 'D'] # назви класів
output_path = "classified_image.jpg" # Зберегти результуюче зображення
# Кольори для позначення кожного класу
# Червоний, Зелений, Блакитний, Пурпуровий, Білий
class_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
class_block_size = 50 # розмір області класифікації
kulbak_lambda = 2

# головна функція класифікації
def startRecognition(big_image, class_images, class_colors, block_size):
    # відкриття та відображення великого зображення у вікні
    image = cv2.imread(big_image)
    # Конвертація в RGB для правильного відображення в matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1)
    ax.set_title("Big image")
    ax.imshow(image)

    # Отримуємо значення кольорів для кожного класу
    pixel_values = extract_pixel_values(class_images)
    # Знаходимо оптимальне значення дельти
    delta = find_delta(pixel_values)

    # ініціалізуємо масив для бінарних матриць
    binary_matrices = []
    # ініціалізуємо масив для еталонних векторів
    etalon_vectors = []
    # Визначаємо значення для масиву контрольних допусків
    control_tolerance_vector = find_control_tolerance_vector(pixel_values[0])

    # Створюємо бінарні матриці для кожного класу та їх еталонні вектори
    for values in pixel_values:
        binary_matrix = create_binary_matrix(values, control_tolerance_vector, delta)
        binary_matrices.append(binary_matrix)
        etalon_vectors.append(create_etalon_vector(binary_matrix))

    # Знаходимо радіуси контейнерів для кожного класу
    radius = find_class_optimal_radius(etalon_vectors, binary_matrices)
    print("Optimal radius for 4 classes:", radius)

    # Починаємо розпізнавати фрагменти
    exam_algorithm(big_image, class_colors, block_size, delta, 
                   control_tolerance_vector, etalon_vectors, radius)

# Функція пошуку оптимальної дельти для СКД
def find_delta(pixel_values):
    # установлюємо початкове значення для delta
    optimal_delta = 0
    # установлюємо початкове значення для критерію
    best_criteria_value = 0
    # Масиви, які будуть використані для побудови графіка
    deltas = []
    criteria_values = []
    criteria_values_work_area = []

    print("Calculation of the optimal delta")

    # Перевіряємо дельту в діапазоні від 1 до 116
    for delta in range(1, 117):
        deltas.append(delta)
        print("For delta " + str(delta))
        # ініціалізуємо масив для бінарних матриць
        binary_matrices = []
        # ініціалізуємо масив для еталонних векторів
        etalon_vectors = []
        # шукаємо значення для масиву контрольних допусків
        control_tolerance_vector = find_control_tolerance_vector(pixel_values[0])

        # Створюємо бінарні матриці для кожного класу та їх еталонні вектори
        for values in pixel_values:
            binary_matrix = create_binary_matrix(values, control_tolerance_vector, delta)
            binary_matrices.append(binary_matrix)
            etalon_vectors.append(create_etalon_vector(binary_matrix))

        # Пошук класів сусідів
        neighbor_pairs = find_neighbors(etalon_vectors)
        # Пошук інформаційного критерію
        criteria_results = compute_criteria(etalon_vectors, binary_matrices, neighbor_pairs)

        # Перевіряємо наявність робочих областей
        working_area_values = [max([pair[0] for pair in result if pair[1]], default=0)
                               for result in criteria_results]

        # Обчислюємо середнє значення критерію та оновлюємо оптимальну дельту
        current_value = np.mean(working_area_values)
        if current_value > best_criteria_value:
            optimal_delta = delta
            best_criteria_value = current_value

        criteria_values.append(np.mean([max([pair[0] for pair in result], default=0)
                              for result in criteria_results]));
        criteria_values_work_area.append(current_value if current_value > 0 else -1);

        print("Delta | criteria value | criteria value in working area")
        print(delta, " | ", np.mean([max([pair[0] for pair in result], default=0)
                              for result in criteria_results]), " | ",
              current_value if current_value > 0 else -1)

    # Побудова графіка
    # Фільтрація значень
    filtered_criteria_values = [cv if cv >= 0 and not np.isinf(cv) else np.nan for cv in criteria_values]
    filtered_criteria_values_work_area = [cva if cva >= 0 and not np.isinf(cva) else np.nan for cva in criteria_values_work_area]

    # Створення маски для збігання значень
    overlap_exact = [
        cv if cv == cva and not (np.isnan(cv) or np.isnan(cva)) else np.nan
        for cv, cva in zip(filtered_criteria_values, filtered_criteria_values_work_area)
    ]

    plt.figure(figsize=(10, 6))
    
    # Заповнення області для Criteria Values
    plt.fill_between(deltas, 0, filtered_criteria_values, 
                     where=[not np.isnan(cv) for cv in filtered_criteria_values], 
                     color='blue', alpha=0.3, label='Kulbak Criteria Values Area')
    
    # Заповнення області для Criteria Values (Working Area)
    plt.fill_between(deltas, 0, filtered_criteria_values_work_area, 
                     where=[not np.isnan(cva) for cva in filtered_criteria_values_work_area], 
                     color='green', alpha=0.3, label='Working Area')

    # Виділення області точного збігання значень
    plt.fill_between(deltas, 0, overlap_exact, 
                     where=[not np.isnan(oe) for oe in overlap_exact], 
                     color='red', alpha=0.5, label='Exact Match Area')
    
    # Лінія для оптимальної delta
    plt.axvline(x=optimal_delta, color='red', linestyle='--', label=f'Optimal Delta = {optimal_delta}')

    plt.xlabel("Delta")
    plt.ylabel("Kulbak Criteria Value")
    plt.title("Dependency of Kulbak Criteria Values on Delta (with Exact Match Area)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Optimal delta: ", optimal_delta)
    return optimal_delta

def find_class_optimal_radius(etalon_vectors, binary_matrices):
    # Пошук класів сусідів
    neighbor_pairs = find_neighbors(etalon_vectors)
    # Пошук інформаційного критерію
    criteria_results = compute_criteria(etalon_vectors, binary_matrices, neighbor_pairs)

    # Масив для збереження оптимальних радіусів для кожного класу
    class_radius = []
    print("Calculation of radius for classes")
    for i, result in enumerate(criteria_results):
        # Масиви для використання в графіку залежності критерію Кульбака від радіусу
        # Для кожного класу
        is_working_area_set = []
        radiuses = []
        criteria_result_set = []

        print("Class ", class_names[i])
        print("Is working area | radius | criteria value")

        best_radius = -1
        best_value = -1
        # Проходимо по всім радіусам та знаходимо оптимальний
        for j, pair in enumerate(result):
            is_working_area_set.append(pair[1])
            radiuses.append(j)
            criteria_result_set.append(pair[0])
            print(pair[1], j, pair[0])
            if pair[1] and pair[0] > best_value:
                best_value = pair[0]
                best_radius = j
        if best_radius == -1:
            best_radius = 1  # Якщо не знайдено, використовуємо мінімальний радіус

        # Основна область критерію
        plt.fill_between(radiuses, criteria_result_set, color="skyblue", alpha=0.4, label="Kulbak Criteria Value")
        plt.plot(radiuses, criteria_result_set, color="blue", label="Kulbak Criteria Value (line)")
        
        # Область робочої зони
        working_radiuses = np.array(radiuses)[is_working_area_set]
        working_criteria = np.array(criteria_result_set)[is_working_area_set]
        plt.fill_between(working_radiuses, working_criteria, color="green", alpha=0.3, label="Working Area")
        
        # Відображаємо оптимальний радіус на графіку
        plt.axvline(x=best_radius, color="red", linestyle="--", label=f"Optimal Radius = {best_radius}")
        
        plt.title(f"Dependency of Kulbak Criteria Values on Radius for Class {class_names[i]}")
        plt.xlabel("Radius")
        plt.ylabel("Kulbak Criteria Value")
        plt.legend()
        plt.grid()
        plt.show()

        class_radius.append(best_radius)

    return class_radius

# Функція екзамену зображення
def exam_algorithm(big_image_path, colors, block_size, delta, control_tolerance_vector, etalon_vectors, radius):
    # Завантаження зображення
    image = cv2.imread(big_image_path)
    image_height, image_width, _ = image.shape

    # Створення копії для відображення результатів класифікації
    output_image = image.copy()

    # Встановлення параметрів шрифту
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    # Проходимо по кожній області (квадрат) в зображенні
    for i in range(0, image_width, block_size):
        for j in range(0, image_height, block_size):
            try:
                # Вирізаємо область та класифікуємо її
                crop = image[j:j + block_size, i:i + block_size]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_values = image_to_array_from_crop(crop)
                crop_binary_matrix = create_binary_matrix(crop_values, control_tolerance_vector, delta)

                class_index = -1
                highest_value = 0

                # Проводимо порівняння області з еталонами кожного класу
                for k, reference_vector in enumerate(etalon_vectors):
                    result = examine_region(reference_vector, radius[k], crop_binary_matrix)
                    if result > highest_value:
                        class_index = k
                        highest_value = result

                # Додаємо текст
                text_position = (i + 3, j + 20)
                # Якщо область класифікована, позначаємо її номером класу та кольором
                if class_index != -1:
                    # У текст додаємо 
                    cv2.putText(output_image, class_names[class_index], text_position, font, font_scale,
                                colors[class_index], font_thickness)

                    # Малюємо прямокутник навколо області
                    top_left = (i + 1, j + 1)
                    bottom_right = (i + block_size - 1, j + block_size - 1)
                    cv2.rectangle(output_image, top_left, bottom_right, colors[class_index], 3)
                else:
                    cv2.putText(output_image, "N/I", text_position, font, font_scale,
                                colors[4], font_thickness)
                    
                    # Малюємо прямокутник навколо області
                    top_left = (i + 1, j + 1)
                    bottom_right = (i + block_size - 1, j + block_size - 1)
                    cv2.rectangle(output_image, top_left, bottom_right, colors[4], 3)

            except Exception as e:
                print(e)

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    # Збереження результатів класифікації
    cv2.imwrite(output_path, output_image)
    print(f"Success! Classified image saved to {output_path}.")

def extract_pixel_values(images):
    # Завантажуємо зображення класів і перетворюємо їх у масиви пікселів
    pixel_values = []
    for image, class_name in zip(images, class_names):
        pixel_values.append(image_to_array(image, class_name))
    return pixel_values

# Функція перетворення зображення у масив пікселів
def image_to_array(image_path, class_name):
    # відкриваємо зображення
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Виводимо кожне зображення класів
    fig, ax = plt.subplots(1)
    ax.set_title("Class " + class_name)
    ax.imshow(image)
    return image_to_array_from_crop(image)

# Функція перетворення об'єкту зображення у масив пікселів (RGB)
def image_to_array_from_crop(image):
    # Отримуємо висоту, ширину та кількість каналів зображення
    height, width, channels = image.shape
    # Перетворюємо на потрібний формат
    return image.reshape(height, width * channels)

def examine_region(etalon_vector, radius, binary_matrix):
    # Порівняння області з еталонним вектором
    if radius == 0:
        return 0  # Уникаємо ділення на нуль
    result = np.mean([1 - (calculate_vector_distance(etalon_vector, binary_row) / radius)
                     for binary_row in binary_matrix])
    return result

# Функція отримання контрольних допусків
def find_control_tolerance_vector(pixel_values):
    return [np.mean([row[i] for row in pixel_values]) for i in range(len(pixel_values[0]))]

# Функція отримання бінарної матриці на основі СКД
def create_binary_matrix(pixel_values, control_tolerance_vector, delta):
    return [
        [1 if control_tolerance_vector[i] - delta <= value <= control_tolerance_vector[i] + delta 
         else 0 for i, value in enumerate(row)]
        for row in pixel_values]

# Функція отримання еталонного вектору
def create_etalon_vector(binary_matrix):
    return [int(np.round(np.mean([row[i] for row in binary_matrix]))) for i in range(len(binary_matrix[0]))]

# Функція пошуку пар сусідніх класів для кожного класу
def find_neighbors(etalon_vectors):
    # Знаходимо пари сусідніх класів для кожного класу
    neighbor_pairs = [[len(etalon_vectors[0]) + 1, len(etalon_vectors[0]) + 1] for _ in etalon_vectors]
    for i, vector_1 in enumerate(etalon_vectors):
        for j, vector_2 in enumerate(etalon_vectors):
            if i != j: # якщо класи відрізняються
                # шукаємо відстань між 2 класами
                distance = calculate_vector_distance(vector_1, vector_2)
                # якщо дистанція є меншою за існуючий результат
                if distance < neighbor_pairs[i][1]:
                    # змінюємо результат на новий
                    neighbor_pairs[i] = [j, distance]
    # виводимо фінальний результат знаходження класів-сусідів
    print("Base class | Neighbor class | Distance")
    for i, (j, distance) in enumerate(neighbor_pairs):
        print(f"{class_names[i]} \t   | {class_names[j]} \t\t    | {distance}")
    return neighbor_pairs

# Функція обчислення відстані між двома векторами
def calculate_vector_distance(vector_1, vector_2):
    return np.sum(np.abs(np.array(vector_1) - np.array(vector_2)))

 # Функція обчислення значення критерію для кожного класу і радіуса
def compute_criteria(etalon_vectors, binary_matrices, neighbor_pairs):
    # Ініціалізуємо масив отриманих критеріїв
    criteria_results = [[] for _ in etalon_vectors]

    for class_number in range(len(etalon_vectors)):
        for radius in range(101):
            # Кодова відстань для базового класу
            sk = [calculate_vector_distance(etalon_vectors[class_number], binary_row) 
                for binary_row in binary_matrices[class_number]]
            # Кодова відстань для класу-сусіда
            sk_para = [calculate_vector_distance(etalon_vectors[class_number], binary_row) 
                for binary_row in binary_matrices[neighbor_pairs[class_number][0]]]
            # Перша достовірність
            d1 = np.mean([1 if distance <= radius else 0 for distance in sk])
            # Помилка другого роду
            beta = np.mean([1 if distance <= radius else 0 for distance in sk_para])
            # Помилка першого роду
            alpha = 1 - d1
            # Друга достовірність
            d2 = 1 - beta
            # пошук значення критерію Кульбака
            criteria_value = calculate_kulbak(alpha, beta, d1, d2) / calculate_kulbak(0, 0, 1, 1)
            # Перевірка, чи лежить отримане значення в межах робочої області
            is_within_working_area = d1 >= 0.5 > beta
            criteria_results[class_number].append((criteria_value, is_within_working_area))

    return criteria_results

# Функція розрахунку критерію Кульбака
def calculate_kulbak(alpha, beta, d1, d2):
    return (0.5 * np.log2((d1 + d2 + 10 ** (-kulbak_lambda)) / (alpha + beta + 10 ** (-kulbak_lambda)))) * ((d1 + d2) - (alpha + beta))

# початок розпізнавання
startRecognition(big_image, image_filenames, class_colors, class_block_size)


