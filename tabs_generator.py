import os
import random
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance





class NotesGenerator:
    numbers_on_lines = False
    dir_name = "tabs"





    def __init__(self):
        for i in range(4):
            notes = self.generate_notes(random.randint(1,10))
            chord_spacing = random.randint(50, 70)
            output_name = f"tab_{i}"
            self.generate_tab(notes, output_name, chord_spacing)





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

        except Exception as excepttion:
            print(f"Error in  generate_notes: {excepttion}")

        return notes





    def generate_tab(self, notes, output_name, chord_spacing=50):
        if not os.path.exists(self.dir_name):  os.makedirs(self.dir_name)
        if chord_spacing < 50:  chord_spacing = 50

        max_x = max(note['x_pos'] for note in notes) if notes else 100
        margin_x = chord_spacing  + random.randint(0, 10)
        width = max_x*chord_spacing  + margin_x + random.randint(0, 10)

        line_spacing = int(chord_spacing/2)  + random.randint(0, 10)
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

        yolo_labels = []



        for note in notes:
            # pobranie danych wejściowych
            string_idx = note['string']  # 1-6
            fret_val = note['fret']  # 1-24
            x = note['x_pos']*chord_spacing

            y = string_y_positions[string_idx - 1]
            text = str(fret_val)

            bbox = draw.textbbox((x, y), text, font=font, anchor="mm")
            if not self.numbers_on_lines:
                draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg_color)
            draw.text((x, y), text, fill=numbers_color, font=font, anchor="mm")

            # etykietowanie
            class_id = fret_val
            w_box = (bbox[2] - bbox[0]) / width
            h_box = (bbox[3] - bbox[1]) / height
            x_center = x / width
            y_center = y / height
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}")



        # Dodanie zakłóceń
        image = self.apply_noise(image, 0.6)

        # Zapis plików
        self.save_file(image, output_name, yolo_labels, notes)





    def apply_noise(self, image, noice_percentage=0.5):
        try:
            # jasność
            if random.random() < noice_percentage:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            # kontrast
            if random.random() < noice_percentage:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            # szum Gaussa
            if random.random() < noice_percentage:
                np_image = np.array(image)
                # Generowanie szumu
                noise = np.random.normal(loc=0, scale=15, size=np_image.shape)
                # Dodanie szumu i zabezpieczenie przed wyjściem poza zakres 0-255
                np_image = np.clip(np_image.astype('int16') + noise, 0, 255).astype('uint8')
                image = Image.fromarray(np_image)

            # rozmycie
            if random.random() < noice_percentage:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            # artefakty kompresji wideo
            if random.random() < noice_percentage:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=random.randint(10, 70))
                buffer.seek(0)
                image = Image.open(buffer).convert('RGB')

        except Exception as excepttion:
            print(f"Error in  apply_noise: {excepttion}")

        return image





    def save_file(self, image, output_name, yolo_labels, notes):
        try:
            output_name = self.dir_name + "\\" + output_name
            image.save(f"{output_name}.png")
            with open(f"{output_name}.txt", "w") as f:
                f.write("\n".join(yolo_labels))

            print(f"Sukces: {output_name} (Klasy: {[n['fret'] for n in notes]})\n\n")

        except Exception as excepttion:
            print(f"Error in  save_file: {excepttion}")





notes_generator = NotesGenerator()