import os
import random
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance





DATA_DIR = '..\\dataset'
# DATA_DIR = '..\\testset'

GEN_NUMBER = 100





class NotesGenerator:





    def __init__(self):
        for i in range(GEN_NUMBER):
            notes = self.generate_notes(random.randint(1,20))
            chord_spacing = random.randint(100, 300)
            line_spacing = random.randint(50, 150)
            numbers_on_lines = random.choice([True, False])
            output_name = f"tab_{i}"
            self.generate_tab(notes, output_name, chord_spacing, line_spacing, numbers_on_lines)





    def generate_notes(self, chord_length):
        notes = []

        try:
            for i in range(chord_length):  # akordy
                chord_number = random.choices([1, 2, 3, 4, 5, 6], [200, 150, 100, 5, 2, 1])[0]
                strings = [1, 2, 3, 4, 5, 6]
                string = random.sample(strings, chord_number)

                for j in range(chord_number):  # chwyty
                    fret= random.randint(1, 24)
                    notes.append({'string': string[j], 'fret': fret, 'x_pos': (i+1)})

            notes.sort(key=lambda x: (x['x_pos'], x['string']))
            print("\n".join(str(n) for n in notes))

        except Exception as e:
            print(f"Error in  generate_notes: {e}")

        return notes





    def generate_tab(self, notes, output_name, chord_spacing=50, line_spacing=50, numbers_on_lines=False):
        max_x = max(note['x_pos'] for note in notes) if notes else 100
        margin_x = chord_spacing  + random.randint(0, 100) + random.randint(0, 100) + random.randint(0, 100)
        width = max_x*chord_spacing  + margin_x + random.randint(0, 10)

        line_spacing = int(line_spacing/2)  + random.randint(0, 10)
        height = line_spacing*8  + random.randint(0, 10)


        # losowanie kolorów tła, lini i czcionki
        bg_color = ( random.randint(240, 255), random.randint(240, 255), random.randint(240, 255) )
        line_color = ( random.randint(0, 10), random.randint(0, 10), random.randint(0, 10) )
        numbers_color = ( random.randint(0, 10), random.randint(0, 10), random.randint(0, 10) )


        image = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(image)

        try:
            fonts = ["arial.ttf", "times.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf", "georgia.ttf", "cour.ttf", "consola.ttf", "impact.ttf", "comic.ttf", "arialbd.ttf"]
            font_size = line_spacing  + random.randint(-5, 3)
            font = ImageFont.truetype(random.choice(fonts), font_size)
        except:
            font = ImageFont.load_default()

        start_y = (height - (5 * line_spacing)) // 2
        string_y_positions = [start_y + i * line_spacing for i in range(6)]

        for y in string_y_positions:
            draw.line([(0, y), (width, y)], fill=line_color, width=2)



        # rysowanie pionowych linii taktu (barlines)
        if max_x > 1:
            current_prob = min(1.0, 0.40 + (max_x * 0.05))

            num_vertical_lines = 0
            while random.random() < current_prob:
                num_vertical_lines += 1
                current_prob /= 1.2

            available_gaps = [int((i + 0.5) * chord_spacing) for i in range(1, max_x)]
            num_vertical_lines = min(num_vertical_lines, len(available_gaps))

            if num_vertical_lines > 0:
                selected_gaps = random.sample(available_gaps, num_vertical_lines)
                y_top = string_y_positions[0]
                y_bottom = string_y_positions[5]

                line_width = random.randint(2, 5)
                for gap_x in selected_gaps:
                    draw.line([(gap_x, y_top), (gap_x, y_bottom)], fill=line_color, width=line_width)



        labels_model_a = []
        chords_model_b = {}
        for note in notes:
            # pobranie danych wejściowych
            string_idx = note['string']-1  # 0-5
            string_num = note['string']  # 1-6
            fret_val = note['fret']  # 1-24
            x = note['x_pos']*chord_spacing

            # rysowanie danych na obrazie
            y = string_y_positions[string_idx]
            text = str(fret_val)
            bbox = draw.textbbox((x, y), text, font=font, anchor="mm")
            if not numbers_on_lines:
                draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg_color)
            draw.text((x, y), text, fill=numbers_color, font=font, anchor="mm")

            # etykietowanie modelu A
            class_id = fret_val
            w_box = (bbox[2] - bbox[0]) / width
            h_box = (bbox[3] - bbox[1]) / height
            x_center = x / width
            y_center = y / height
            labels_model_a.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}")

            # etykietowanie modelu B
            if x not in chords_model_b:
                chords_model_b[x] = []
            chords_model_b[x].append(f"{string_num}:{fret_val}")

        # etykietowanie modelu B
        sorted_chords_model_b = sorted(chords_model_b.keys())
        sequence_steps_model_b = []
        for x in sorted_chords_model_b:
            step_str = ",".join(str(fret) for fret in chords_model_b[x])
            sequence_steps_model_b.append(step_str)
        labels_model_b = " | ".join(sequence_steps_model_b)



        # Dodanie zakłóceń
        image = self.apply_noise(image, 0.6)

        # Zapis plików
        self.save_files(image, output_name, labels_model_a, labels_model_b, notes)





    def apply_noise(self, image, noise_percentage=0.5):
        try:
            # jasność
            if random.random() < noise_percentage:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            # kontrast
            if random.random() < noise_percentage:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            # szum Gaussa
            if random.random() < noise_percentage:
                np_image = np.array(image)
                noise = np.random.normal(loc=0, scale=15, size=np_image.shape)
                np_image = np.clip(np_image.astype('int16') + noise, 0, 255).astype('uint8')
                image = Image.fromarray(np_image)

            # motion blur
            if random.random() < noise_percentage:
                np_image = np.array(image)
                kernel_size = random.choice([1, 3, 5, 7])
                kernel_motion_blur = np.zeros((kernel_size, kernel_size))
                kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                kernel_motion_blur /= kernel_size
                np_image = cv2.filter2D(np_image, -1, kernel_motion_blur)
                image = Image.fromarray(np_image)

            # rozmycie
            if random.random() < noise_percentage:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            # artefakty kompresji wideo
            if random.random() < noise_percentage:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=random.randint(10, 70))
                buffer.seek(0)
                image = Image.open(buffer).convert('RGB')

        except Exception as e:
            print(f"Error in  apply_noise: {e}")

        return image





    def save_files(self, image, output_name, labels_model_a, labels_model_b, notes):
        try:
            if not os.path.exists(DATA_DIR):  os.makedirs(DATA_DIR)
            # if not os.path.exists(IMAGE_DIR):  os.makedirs(IMAGE_DIR)
            # if not os.path.exists(LABEL_DIR_A):  os.makedirs(LABEL_DIR_A)
            # if not os.path.exists(LABEL_DIR_B):  os.makedirs(LABEL_DIR_B)

            image.save(f"{DATA_DIR}\\{output_name}.png")
            with open(f"{DATA_DIR}\\{output_name}_a.txt", "w") as f:  f.write("\n".join(labels_model_a))
            with open(f"{DATA_DIR}\\{output_name}_b.txt", "w") as f:  f.write(labels_model_b)

            print(f"Sukces: {output_name} (Klasy: {[n['fret'] for n in notes]})\n\n")

        except Exception as e:
            print(f"Error in  save_files: {e}")





if __name__ == "__main__":
    notes_generator = NotesGenerator()