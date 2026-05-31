import os
import random
import io
import cv2
import gc
import numpy as np
import concurrent.futures
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance



DATA_DIR = '..\\dataset'
# DATA_DIR = '..\\testset'

GEN_NUMBER = 100



class NotesGenerator:

    def __init__(self):
        pass



    def generate_single_tab(self, index):
        line_spacing = random.randint(30, 80)
        chord_spacing = int(line_spacing * random.uniform(1.3, 2.5))

        max_allowed_width = 1000 * random.uniform(0.4, 2.0)
        notes = []
        current_x = 1

        while current_x * chord_spacing < max_allowed_width:
            chord_number = random.choices([1, 2, 3, 4, 5, 6], [200, 60, 15, 5, 2, 1])[0]
            strings = random.sample([1, 2, 3, 4, 5, 6], chord_number)

            for string in strings:
                fret_weights = [15, 12, 12, 12, 10, 10, 8, 8, 6, 6] + [1] * 15
                fret = random.choices(list(range(25)), weights=fret_weights)[0]
                notes.append({'string': string, 'fret': fret, 'x_pos': current_x})

            current_x += 1
            if current_x > 25: break

        notes.sort(key=lambda x: (x['x_pos'], x['string']))

        numbers_on_lines = random.random() < 0.35
        output_name = f"tab_{index}"

        self.generate_tab(notes, output_name, chord_spacing, line_spacing, numbers_on_lines)

        gc.collect()
        return True



    def generate_tab(self, notes, output_name, chord_spacing=50, line_spacing=50, numbers_on_lines=False):
        max_x = max(note['x_pos'] for note in notes) if notes else 100
        margin_x = chord_spacing + random.randint(0, 100)
        width = int(max_x * chord_spacing + margin_x + random.randint(0, 10))

        strict_tab_height = line_spacing * 5

        top_margin = int(line_spacing * random.uniform(4.0, 6.0))
        bottom_margin = int(line_spacing * random.uniform(3.5, 5.5))
        height = int(strict_tab_height + top_margin + bottom_margin)

        if random.random() < 0.7:
            bg_color = (random.randint(245, 255), random.randint(245, 255), random.randint(245, 255))
        else:
            bg_color = (random.randint(230, 250), random.randint(230, 250), random.randint(230, 250))

        line_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        numbers_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))

        fonts = ["arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf", "tahoma.ttf", "tahomabd.ttf", "verdana.ttf",
                 "verdanab.ttf", "consola.ttf", "consolab.ttf"]

        try:
            font_size = int(line_spacing) + random.randint(-2, 2)
            font = ImageFont.truetype(random.choice(fonts), font_size)
        except:
            font = ImageFont.truetype("arial.ttf", int(line_spacing))

        try:
            measure_font_size = max(10, int(line_spacing * random.uniform(0.4, 0.7)))
            measure_font = ImageFont.truetype(random.choice(fonts), measure_font_size)
        except:
            measure_font = ImageFont.truetype("arial.ttf", 12)

        measure_color = (random.randint(50, 180), random.randint(50, 180), random.randint(50, 180))

        try:
            art_font_size = max(12, int(line_spacing * random.uniform(0.7, 1.4)))
            art_font = ImageFont.truetype(random.choice(fonts), art_font_size)
            small_art_font = ImageFont.truetype(random.choice(fonts), max(10, int(art_font_size * 0.65)))
        except:
            art_font_size = 16
            art_font = ImageFont.truetype("arial.ttf", art_font_size)
            small_art_font = ImageFont.truetype("arial.ttf", 10)

        string_y_positions = [top_margin + i * line_spacing for i in range(6)]
        line_width = random.randint(1, 3)

        base_img = Image.new('RGBA', (width, height), color=bg_color + (255,))

        lines_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_lines = ImageDraw.Draw(lines_img)

        marker_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_marker = ImageDraw.Draw(marker_img)

        for y in string_y_positions:
            draw_lines.line([(0, y), (width, y)], fill=line_color + (255,), width=line_width)

        measure_counter = random.randint(1, 200)
        barline_x_positions = []
        if max_x > 1:
            current_prob = min(1.0, 0.20 + (max_x * 0.02))  # Zmniejszono prawdopodobieństwo kresek taktów
            num_vertical_lines = 0
            while random.random() < current_prob:
                num_vertical_lines += 1
                current_prob /= 1.2

            available_gaps = [int((i + 0.5) * chord_spacing) for i in range(1, max_x)]
            num_vertical_lines = min(num_vertical_lines, len(available_gaps))

            if num_vertical_lines > 0:
                selected_gaps = sorted(random.sample(available_gaps, num_vertical_lines))
                y_top = string_y_positions[0]
                y_bottom = string_y_positions[5]

                for gap_x in selected_gaps:
                    if random.random() < 0.2:
                        draw_lines.line([(gap_x, y_top), (gap_x, y_bottom)], fill=line_color + (255,), width=line_width)
                        draw_lines.line([(gap_x + 6, y_top), (gap_x + 6, y_bottom)], fill=line_color + (255,),
                                        width=line_width + 1)
                    else:
                        draw_lines.line([(gap_x, y_top), (gap_x, y_bottom)], fill=line_color + (255,),
                                        width=line_width + random.randint(-1, 2))
                    barline_x_positions.append(gap_x)

        for note in notes:
            string_idx = note['string'] - 1
            x = note['x_pos'] * chord_spacing

            y_offset = random.randint(-2, 2)
            y = string_y_positions[string_idx] + y_offset
            note['render_y'] = y

            text = str(note['fret'])
            bbox = draw_lines.textbbox((x, y), text, font=font, anchor="mm")
            note['bbox'] = bbox

            if not numbers_on_lines:
                pad_box = (int(bbox[0] - 2), int(bbox[1] - 2), int(bbox[2] + 2), int(bbox[3] + 2))
                lines_img.paste((0, 0, 0, 0), pad_box)

        marker_over_strings = random.random() < 0.5

        # Zmniejszono prawdopodobieństwo markerów (z 0.4 na 0.15)
        if random.random() < 0.15:
            num_highlights = random.randint(1, 2)
            for _ in range(num_highlights):
                hx_start = random.randint(int(chord_spacing), int(width * 0.7))
                hx_end = hx_start + random.randint(int(chord_spacing), int(chord_spacing * 4))
                highlight_colors = [(255, 245, 200, 130), (255, 230, 210, 130), (220, 240, 255, 130)]
                h_color = random.choice(highlight_colors)

                if random.random() < 0.5:
                    hy_start, hy_end = 0, height
                else:
                    hy_start = top_margin - line_spacing
                    hy_end = height - bottom_margin + line_spacing

                draw_marker.rectangle([hx_start, hy_start, hx_end, hy_end], fill=h_color)

        if marker_over_strings:
            final_img = Image.alpha_composite(base_img, lines_img)
            final_img = Image.alpha_composite(final_img, marker_img)
        else:
            final_img = Image.alpha_composite(base_img, marker_img)
            final_img = Image.alpha_composite(final_img, lines_img)

        final_img = final_img.convert('RGB')
        draw_final = ImageDraw.Draw(final_img)

        y_top = string_y_positions[0]

        # Numery taktów
        for gap_x in barline_x_positions:
            # Zmniejszono prawdopodobieństwo numerów taktów (z 0.8 na 0.4)
            if random.random() < 0.4:
                offset_x = random.randint(-30, 0)
                offset_y = random.randint(-10, 5)
                draw_final.text((gap_x + offset_x, y_top + offset_y - line_spacing * 1.1), str(measure_counter),
                                fill=measure_color, font=measure_font)
                measure_counter += 1

        grouped_notes = {}
        for note in notes:
            x = note['x_pos'] * chord_spacing
            if x not in grouped_notes:
                grouped_notes[x] = []
            grouped_notes[x].append(note)

        for x, group in grouped_notes.items():
            # Rysowanie laseczek rytmicznych (z 0.65 na 0.3)
            if random.random() < 0.30:
                bottom_string_idx = max(n['string'] for n in group) - 1
                y_start = string_y_positions[bottom_string_idx] + line_spacing * 0.8

                y_end = string_y_positions[5] + line_spacing * random.uniform(1.5, 2.7)
                y_end = min(y_end, height - random.randint(5, 15))

                draw_final.line([(x, y_start), (x, y_end)], fill=line_color, width=max(1, line_width - 1))

                if random.random() < 0.6:
                    beam_length = chord_spacing * random.uniform(0.6, 1.0)
                    beam_thickness = random.randint(3, 7)
                    draw_final.line([(x, y_end), (x + beam_length, y_end)], fill=line_color, width=beam_thickness)
                else:
                    draw_final.arc([x, y_end - 15, x + 20, y_end + 15], start=180, end=270, fill=line_color,
                                   width=max(1, line_width))

            # Artykulacja na górze (z 0.4 na 0.15)
            if random.random() < 0.15:
                art_text = random.choice(
                    ["P.M.", "H", "P", "C#5", "C5", "D5", "(E5)", "Dsus2", "Cmaj7", "B7sus4", "1/4", "1/2", "Full",
                     "1 1/2", "PB", "T", "S", "sl.", "v", "vib", "Tr.", "harm.", "N.H.", "A.H.", "P.H.", "T.H.",
                     "w/bar"])

                y_art = y_top - line_spacing * random.uniform(1.8, 2.8)
                y_art = max(y_art, art_font_size + random.randint(5, 15))

                current_art_font = art_font if "5" in art_text else small_art_font
                draw_final.text((x, y_art), art_text, fill=numbers_color, font=current_art_font, anchor="md")

                # STRZAŁKI BENDÓW
                if any(bend_type in art_text for bend_type in ["1/", "Full", "PB"]):
                    top_note = min(group, key=lambda n: n['render_y'])
                    x0 = top_note['bbox'][2] + 2
                    y0 = top_note['render_y']

                    x1 = x + line_spacing * 0.2
                    y1 = y_art + 10

                    cx = x0 + line_spacing * 0.5
                    cy = y0 - line_spacing * 0.2

                    pts = []
                    for t in np.linspace(0, 1, 15):
                        xt = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
                        yt = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
                        pts.append((xt, yt))

                    draw_final.line(pts, fill=line_color, width=max(1, line_width))

                    aw = line_spacing * 0.12
                    ah = line_spacing * 0.15
                    draw_final.polygon([(x1 - aw, y1 + ah), (x1, y1 - 2), (x1 + aw, y1 + ah - 2)], fill=line_color)

            # Łuki na dole (z 0.25 na 0.1)
            if random.random() < 0.10:
                y_arc = string_y_positions[max(n['string'] for n in group) - 1] + line_spacing * 0.5
                draw_final.arc([x, y_arc, x + chord_spacing * 0.9, y_arc + line_spacing], start=0, end=180,
                               fill=line_color, width=max(1, line_width))

        # RYSOWANIE NUT NA SAMYM WIERZCHU ORAZ ETYKIETOWANIE + SLIDES
        labels_model_a = []
        chords_model_b = {}
        stroke_width = random.choices([0, 1], weights=[0.3, 0.7])[0]

        for note in notes:
            x = note['x_pos'] * chord_spacing
            y = note['render_y']
            text = str(note['fret'])
            bbox = note['bbox']
            string_num = note['string']

            # Rysowanie samej cyfry
            draw_final.text((x, y), text, fill=numbers_color, font=font, anchor="mm", stroke_width=stroke_width,
                            stroke_fill=numbers_color)

            # Rysowanie ukośnych kresek - zmniejszono z 0.12 na 0.05
            if random.random() < 0.05:  # Slide In (/)
                draw_final.line([
                    (bbox[0] - line_spacing * 0.3, y + line_spacing * 0.25),
                    (bbox[0] - line_spacing * 0.05, y - line_spacing * 0.25)
                ], fill=line_color, width=max(1, line_width))
            elif random.random() < 0.05:  # Slide Out (\)
                draw_final.line([
                    (bbox[2] + line_spacing * 0.05, y - line_spacing * 0.25),
                    (bbox[2] + line_spacing * 0.3, y + line_spacing * 0.25)
                ], fill=line_color, width=max(1, line_width))

            # Etykietowanie modelu A
            class_id = note['fret']
            w_box = (bbox[2] - bbox[0]) / width
            h_box = (bbox[3] - bbox[1]) / height
            x_center = x / width
            y_center = y / height
            labels_model_a.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}")

            # Etykietowanie CRNN (model B)
            if x not in chords_model_b:
                chords_model_b[x] = []
            chords_model_b[x].append(f"{string_num}:{note['fret']}")

        sorted_chords_model_b = sorted(chords_model_b.keys())
        sequence_steps_model_b = []
        for x in sorted_chords_model_b:
            step_str = ",".join(str(fret) for fret in chords_model_b[x])
            sequence_steps_model_b.append(step_str)
        labels_model_b = " | ".join(sequence_steps_model_b)

        # Dodanie zakłóceń i zapis
        final_img = self.apply_noise(final_img, 0.6)
        self.save_files(final_img, output_name, labels_model_a, labels_model_b)



    def apply_noise(self, image, noise_percentage=0.5):
        try:
            if random.random() < noise_percentage:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            if random.random() < noise_percentage:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            if random.random() < noise_percentage:
                np_image = np.array(image)
                noise = np.random.normal(loc=0, scale=15, size=np_image.shape)
                np_image = np.clip(np_image.astype('int16') + noise, 0, 255).astype('uint8')
                image = Image.fromarray(np_image)

            if random.random() < noise_percentage:
                np_image = np.array(image)
                kernel_size = random.choice([1, 3, 5, 7])
                kernel_motion_blur = np.zeros((kernel_size, kernel_size))
                kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                kernel_motion_blur /= kernel_size
                np_image = cv2.filter2D(np_image, -1, kernel_motion_blur)
                image = Image.fromarray(np_image)

            if random.random() < noise_percentage:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            if random.random() < noise_percentage:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=random.randint(10, 70))
                buffer.seek(0)
                image = Image.open(buffer).convert('RGB')

        except Exception as e:
            print(f"Error in apply_noise: {e}")

        return image



    def save_files(self, image, output_name, labels_model_a, labels_model_b):
        try:
            image.save(f"{DATA_DIR}\\{output_name}.png")
            with open(f"{DATA_DIR}\\{output_name}_a.txt", "w") as f:
                f.write("\n".join(labels_model_a))
            with open(f"{DATA_DIR}\\{output_name}_b.txt", "w") as f:
                f.write(labels_model_b)
        except Exception as e:
            print(f"Error in save_files: {e}")





if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    generator = NotesGenerator()

    print(f"Rozpoczynamy generowanie {GEN_NUMBER} obrazków w puli procesów...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(generator.generate_single_tab, range(GEN_NUMBER)), total=GEN_NUMBER))

    print("\nGenerowanie zakończone!")