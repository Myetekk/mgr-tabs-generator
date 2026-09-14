import os
import random
import io
import cv2
import gc
import numpy as np
import concurrent.futures
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

GEN_NUMBER = 1000


class NotesGenerator:



    def __init__(self, data_dir, noise_percentage=0.6, add_elem_percentage=0.8, time_elem_percentage=0.4):
        self.line_spacing = None
        self.chord_spacing = None
        self.notes = []

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.noise_percentage = noise_percentage
        self.add_elem_percentage = add_elem_percentage
        self.time_elem_percentage = time_elem_percentage



    def generate_single_tab(self, index):
        """
        Generates the mathematical positions and properties for notes on a single tablature snippet,
        then triggers the rendering process.
        """
        self.notes = []

        if random.random() < 0.96:  self.line_spacing = random.randint(30, 80)  # normal line spacing
        else:  self.line_spacing = random.randint(20, 30)  # small line spacing in 4% chance

        if random.random() < 0.96:  self.chord_spacing = int(self.line_spacing * random.uniform(1.3, 2.5))  # normal chord spacing
        else:  self.chord_spacing = int(self.line_spacing * random.uniform(1.1, 1.3))  # small chord spacing in 4% chance

        if random.random() < 0.98:  max_allowed_width = random.randint(int(self.chord_spacing * 2), 1000)  # normal tab width
        else:  max_allowed_width = random.randint(int(self.chord_spacing * 1.1), int(self.chord_spacing * 1.9))  # very narrow tab in 2% chance

        # Populate the tablature with random chords and single notes
        self.has_time_sig = random.random() < 0.2
        current_x = 2 if self.has_time_sig else 1
        self.empty_columns_for_rests = []

        # Ensure max_allowed_width allows at least one note to be generated
        if max_allowed_width <= current_x * self.chord_spacing:
            max_allowed_width = int((current_x + 1.5) * self.chord_spacing)

        while current_x * self.chord_spacing < max_allowed_width:
            if random.random() < 0.15:  # 15% chance to leave a gap for a rest
                self.empty_columns_for_rests.append(current_x)
                current_x += 1
                if current_x > 25: break
                continue

            chord_number = random.choices([1, 2, 3, 4, 5, 6], [200, 60, 15, 5, 2, 1])[0]
            strings = random.sample([1, 2, 3, 4, 5, 6], chord_number)

            for string in strings:
                # Weighted random distribution of frets (lower frets are more common)
                fret_choices = list(range(25)) + ['X']
                fret_weights = [15, 12, 12, 12, 10, 10, 8, 8, 6, 6] + [1] * 15 + [3]
                fret = random.choices(fret_choices, weights=fret_weights)[0]
                self.notes.append({'string': string, 'fret': fret, 'x_pos': current_x})

            current_x += 1
            if current_x > 25: break

        self.notes.sort(key=lambda x: (x['x_pos'], x['string']))

        output_name = f"tab_{index}"
        self.generate_tab(output_name)

        gc.collect()
        return True



    def generate_tab(self, output_name):
        """
        Orchestrates the drawing of the tablature image layer by layer.
        Refactored to delegate specific visual components to private helper methods.
        """

        # Set basic settings
        self._general_settings()

        # Calculate dimensions
        all_x_positions = [note['x_pos'] for note in self.notes] + getattr(self, 'empty_columns_for_rests', [])
        if getattr(self, 'has_time_sig', False):
            all_x_positions.append(1)
            
        max_x = max(all_x_positions) if all_x_positions else 2
        margin_x = self.chord_spacing + random.randint(0, 100)
        width = int(max_x * self.chord_spacing + margin_x + random.randint(0, 10))

        strict_tab_height = self.line_spacing * 5
        top_margin = int(self.line_spacing * random.uniform(1.0, 3.0))
        bottom_margin = int(self.line_spacing * random.uniform(1.0, 3.0))
        height = int(strict_tab_height + top_margin + bottom_margin)
        string_y_positions = [top_margin + i * self.line_spacing for i in range(6)]

        # Base Styling (Colors)
        self._generate_color_palette()

        # Image layers setup
        base_img = Image.new('RGBA', (width, height), color=self.bg_color + (255,))
        lines_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_lines = ImageDraw.Draw(lines_img)
        marker_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_marker = ImageDraw.Draw(marker_img)

        # Draw horizontal staff strings and vertical measure barlines
        barline_x_positions = self._draw_staff_and_barlines(draw_lines, width, string_y_positions, max_x)

        # Pre-calculate note bounding boxes and optionally clear background lines
        self._prepare_note_bboxes(draw_lines, string_y_positions, lines_img)

        # Draw random highlight markers over the background
        self._draw_highlight_markers(draw_marker, width, height, top_margin, bottom_margin)

        # Composite the base layers based on transparency priorities
        # Strings over marker
        final_img = Image.alpha_composite(base_img, marker_img)
        final_img = Image.alpha_composite(final_img, lines_img)

        final_img = final_img.convert('RGB')
        draw_final = ImageDraw.Draw(final_img)

        # Draw measure numbers above the barlines
        self._draw_measure_numbers(draw_final, barline_x_positions, string_y_positions[0])

        # Group notes by X-coordinate for vertical alignment drawing
        grouped_notes = {}
        for note in self.notes:
            grouped_notes.setdefault(note['x_pos'] * self.chord_spacing, []).append(note)

        # Draw structural groupings and articulations
        self._draw_beams(draw_final, grouped_notes, string_y_positions, height)
        self._draw_articulations(draw_final, grouped_notes, string_y_positions)
        self._draw_bottom_arcs(draw_final)

        # Draw time signatures and rests (NOISE for the model to ignore)
        self._draw_time_signatures(draw_final, string_y_positions)
        self._draw_rests(draw_final, string_y_positions)

        # Draw the "current time" vertical line
        self._draw_time_line(draw_final, max_x, string_y_positions)

        # Draw the actual note text and generate the label sequences
        stroke_width = random.choices([0, 1], weights=[0.3, 0.7])[0]

        labels_model_a, labels_model_b = self._draw_notes_and_get_labels(draw_final, width, height, stroke_width)

        # Apply visual distortions and save files
        final_img = self._apply_noise(final_img)
        self._save_files(final_img, output_name, labels_model_a, labels_model_b)



    def _general_settings(self):
        """Settings that can be applied to multiple functions"""
        self.numbers_on_lines = random.random() < 0.35

        self.line_width = random.randint(1, 4)

        self.fonts = ["arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf", "tahoma.ttf", "tahomabd.ttf", "verdana.ttf", "verdanab.ttf", "consola.ttf", "consolab.ttf"]
        self.font_size = int(self.line_spacing) + random.randint(-5, 5)
        try:
            self.font = ImageFont.truetype(random.choice(self.fonts), self.font_size)
        except:
            self.font = ImageFont.truetype("arial.ttf", self.font_size)



    def _generate_color_palette(self):
        """Generates random RGB colors with slight variations for different UI elements."""
        bg_range = random.randint(230, 255)
        self.bg_color = (bg_range + random.randint(-5, 5), bg_range + random.randint(-5, 5), bg_range + random.randint(-5, 5))

        line_range = random.randint(0, 70)
        self.line_color = (line_range + random.randint(-5, 5), line_range + random.randint(-5, 5), line_range + random.randint(-5, 5))

        num_range = random.randint(0, 40)
        self.numbers_color = (num_range + random.randint(-5, 5), num_range + random.randint(-5, 5), num_range + random.randint(-5, 5))



    def _draw_staff_and_barlines(self, draw_lines, width, string_y_positions, max_x):
        """
        Draws the standard 6 horizontal strings. Also generates random vertical lines
        that separate the tablature into measures, occasionally using double barlines.
        Returns the X coordinates of these barlines for measure numbering.
        """
        for y in string_y_positions:
            draw_lines.line([(0, y), (width, y)], fill=self.line_color + (255,), width=self.line_width)

        barline_x_positions = []
        if max_x > 1:
            current_prob = min(1.0, 0.20 + (max_x * 0.02))
            num_vertical_lines = 0
            while random.random() < current_prob:
                num_vertical_lines += 1
                current_prob /= 1.2

            if random.random() < self.time_elem_percentage:
                available_gaps = [int((i + 0.5) * self.chord_spacing) for i in range(1, max_x)]
                num_vertical_lines = min(num_vertical_lines, len(available_gaps))

                if num_vertical_lines > 0:
                    selected_gaps = sorted(random.sample(available_gaps, num_vertical_lines))
                    y_top, y_bottom = string_y_positions[0], string_y_positions[5]

                    for gap_x in selected_gaps:
                        if random.random() < 0.2:
                            # Double barline: thin line + thick line
                            thin_w = max(1, self.line_width)
                            thick_w = thin_w + random.randint(1, 3)
                            gap = thin_w + thick_w + random.randint(1, 3)
                            draw_lines.line([(gap_x, y_top), (gap_x, y_bottom)], fill=self.line_color + (255,), width=thin_w)
                            draw_lines.line([(gap_x + gap, y_top), (gap_x + gap, y_bottom)], fill=self.line_color + (255,), width=thick_w)
                        else:
                            draw_lines.line([(gap_x, y_top), (gap_x, y_bottom)], fill=self.line_color + (255,), width=self.line_width + random.randint(-1, 2))
                        barline_x_positions.append(gap_x)

        return barline_x_positions



    def _prepare_note_bboxes(self, draw_lines, string_y_positions, lines_img):
        """Calculates bounding boxes for text and masks out the background lines if needed."""
        for note in self.notes:
            string_idx = note['string'] - 1
            x = note['x_pos'] * self.chord_spacing
            y = string_y_positions[string_idx] + random.randint(-2, 2)
            note['render_y'] = y

            bbox = draw_lines.textbbox((x, y), str(note['fret']), font=self.font, anchor="mm")
            note['bbox'] = bbox

            if not self.numbers_on_lines:
                pad_box = (int(bbox[0] - 2), int(bbox[1] - 2), int(bbox[2] + 2), int(bbox[3] + 2))
                lines_img.paste((0, 0, 0, 0), pad_box)



    def _draw_highlight_markers(self, draw_marker, width, height, top_margin, bottom_margin):
        """Draws transparent highlight rectangles (markers) typical in scanned/annotated documents."""
        if random.random() > self.time_elem_percentage/2: return

        num_highlights = random.randint(1, 2)
        for _ in range(num_highlights):
            hx_start = random.randint(int(self.chord_spacing), int(width * 0.7))
            hx_end = hx_start + random.randint(int(self.chord_spacing), int(self.chord_spacing * 5))

            rand_color = random.random()
            if rand_color < 0.4:
                h_color = [246+random.randint(-5,5), 247+random.randint(-5,5), 213+random.randint(-5,5)]
            elif rand_color < 0.7:
                h_color = [220 + random.randint(-5, 5), 230 + random.randint(-5, 5), 245 + random.randint(-5, 5)]
            else:
                h_color = [243+random.randint(-5,5), 232+random.randint(-5,5), 223+random.randint(-5,5)]

            if random.random() < 0.5:
                hy_start, hy_end = 0, height
            else:
                hy_start = top_margin - self.line_spacing * random.uniform(0, 3)
                hy_end = height - bottom_margin + self.line_spacing * random.uniform(0, 3)

            draw_marker.rectangle([hx_start, hy_start, hx_end, hy_end], fill=tuple(h_color))



    def _draw_measure_numbers(self, draw_final, barline_x_positions, y_top):
        """Randomly labels measures above vertical barlines."""
        measure_counter = random.randint(1, 200)

        measure_range = random.randint(0, 40)
        measure_color = (measure_range + random.randint(-5, 5), measure_range + random.randint(-5, 5), measure_range + random.randint(-5, 5))

        m_size = self.font_size * random.uniform(0.3, 0.6)
        try:
            measure_font = ImageFont.truetype(random.choice(self.fonts), m_size)
        except:
            measure_font = ImageFont.truetype("arial.ttf", m_size)

        for gap_x in barline_x_positions:
            if random.random() > self.time_elem_percentage: return

            offset_x, offset_y = random.randint(-30, 0), random.randint(-10, 15)
            draw_final.text((gap_x + offset_x, y_top + offset_y - self.line_spacing * 1.1), str(measure_counter), fill=measure_color, font=measure_font)
            measure_counter += 1



    def _draw_beams(self, draw_final, grouped_notes, string_y_positions, height):
        """
        Grouped beam rendering. Walks through sorted X positions and connects adjacent
        note groups with vertical stems and horizontal bottom beams, resembling sheet music rhythm notation.
        """
        if random.random() > self.add_elem_percentage: return

        sorted_x_positions = sorted(grouped_notes.keys())
        beam_color = tuple(max(0, min(255, c + random.randint(-3, 3))) for c in self.line_color)

        def draw_beam_group(bg):
            """Draw stems + shared beam bar for a list of x positions."""
            beam_width = max(1, 5)

            if len(bg) < 2:  # single beam with arc
                x = bg[0]
                grp = grouped_notes[x]
                bottom_string_idx = max(n['string'] for n in grp) - 1
                y_start = string_y_positions[bottom_string_idx] + self.line_spacing * random.uniform(0.25, 0.7)
                y_end = string_y_positions[5] + self.line_spacing * random.uniform(1.2, 3.5)

                draw_final.line([(x, y_start), (x, y_end)], fill=beam_color, width=beam_width)

                # Dynamic curve at the bottom (eighth-note flag)
                arc_w = int(self.line_spacing * random.uniform(0.4, 1.0))
                arc_h = int(self.line_spacing * random.uniform(0.4, 1.1))
                draw_final.arc([x, y_end - arc_h, x + arc_w, y_end + arc_h], start=180, end=270, fill=beam_color, width=beam_width+random.randint(-1,1))
            else:  # conected beams
                beam_y = string_y_positions[5] + self.line_spacing * random.uniform(1.0, 4.5)
                beam_thickness = beam_width + random.randint(-1, 3)
                stem_x_list = []

                for x in bg:
                    grp = grouped_notes[x]
                    bottom_string_idx = max(n['string'] for n in grp) - 1
                    y_start = string_y_positions[bottom_string_idx] + self.line_spacing * random.uniform(0.25, 0.7)
                    draw_final.line([(x, y_start), (x, beam_y)], fill=beam_color, width=beam_width)
                    stem_x_list.append(x)

                draw_final.line([(stem_x_list[0], beam_y), (stem_x_list[-1], beam_y)], fill=beam_color, width=beam_thickness)

                if len(bg) >= 4 and random.random() < 0.4:
                    second_beam_y = beam_y - beam_thickness - random.randint(2, 5)
                    mid = len(stem_x_list) // 2
                    draw_final.line([(stem_x_list[0], second_beam_y), (stem_x_list[mid - 1], second_beam_y)], fill=beam_color, width=beam_thickness)

        i = 0
        while i < len(sorted_x_positions):
            if random.random() < 0.30:
                max_size = min(6, len(sorted_x_positions) - i)
                group_size = random.randint(2, max(2, max_size))
                candidate = sorted_x_positions[i:i + group_size]
                bg = [candidate[0]]

                # Expand group if notes are close enough
                for xi in candidate[1:]:
                    if xi - bg[-1] <= self.chord_spacing * 1.6:
                        bg.append(xi)
                    else:
                        break
                draw_beam_group(bg)
                i += len(bg)
            else:
                i += 1



    def _draw_articulations(self, draw_final, grouped_notes, string_y_positions):
        """Draws articulation marks (e.g. Palm Mutes, Harmonics, Bend arrows) above the staff."""
        if random.random() > self.add_elem_percentage: return

        y_top = string_y_positions[0]
        articulations_color = tuple(max(0, min(255, c + random.randint(-3, 3))) for c in self.line_color)

        for x, group in grouped_notes.items():
            if random.random() > 0.15: return

            art_text = random.choice([
                "P.M.", "H", "P", "C#5", "C5", "D5", "(E5)", "Dsus2", "Cmaj7", "B7sus4", "1/4", "1/2", "Full", "1 1/2", "PB", "T", "S", "sl.", "v", "vib", "Tr.", "harm.", "N.H.", "A.H.", "P.H.", "T.H.", "w/bar"
            ])

            a_size = self.font_size * random.uniform(0.5, 1.3)
            try:
                art_font = ImageFont.truetype(random.choice(self.fonts), a_size)
            except:
                art_font = ImageFont.truetype("arial.ttf", a_size)

            x_art = x + self.chord_spacing * random.uniform(0.0, 1.0)
            y_art = y_top - self.line_spacing * random.uniform(0.5, 5.5)

            # P.M. optionally gets a trailing dashed line (e.g., "P.M. - - - -|")
            if art_text == "P.M." and random.random() < 0.5:
                num_dashes = random.randint(2, 6)
                art_text = "P.M. " + "- " * num_dashes + "|"

            draw_final.text((x_art, y_art), art_text, fill=self.numbers_color, font=art_font, anchor="md")

            # Bend curves and arrows
            if any(bend_type in art_text for bend_type in ["1/", "Full", "PB"]):
                top_note = min(group, key=lambda n: n['render_y'])
                x0, y0 = top_note['bbox'][2] + 2, top_note['render_y']
                x1, y1 = x_art + self.line_spacing * 0.2, y_art + 10
                cx, cy = x0 + self.line_spacing * 0.5, y0 - self.line_spacing * 0.2

                pts = []
                for t in np.linspace(0, 1, 15):
                    xt = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
                    yt = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
                    pts.append((xt, yt))

                draw_final.line(pts, fill=articulations_color, width=max(1, self.line_width))

                # Arrow head
                aw, ah = self.line_spacing * 0.12, self.line_spacing * 0.15
                draw_final.polygon([(x1 - aw, y1 + ah), (x1, y1 - 2), (x1 + aw, y1 + ah - 2)], fill=articulations_color, width=max(1, self.line_width + random.randint(-1, 2)))



    def _draw_bottom_arcs(self, draw_final):
        """
        Draws bottom arcs (hammer-on / pull-off ties). Only connects two notes that are
        on the SAME string and where there is NO other note on ANY string between their positions.
        """
        if random.random() > self.add_elem_percentage: return

        arc_width = max(1, 5)
        all_occupied_x = set(n['x_pos'] for n in self.notes)
        bottom_arcs_color = tuple(max(0, min(255, c + random.randint(-3, 3))) for c in self.line_color)

        notes_by_string = {}
        for note in self.notes:
            notes_by_string.setdefault(note['string'], []).append(note)

        for s in notes_by_string:
            notes_by_string[s].sort(key=lambda n: n['x_pos'])

        for s, string_notes in notes_by_string.items():
            for k in range(len(string_notes) - 1):
                left_note, right_note = string_notes[k], string_notes[k + 1]
                lx, rx = left_note['x_pos'], right_note['x_pos']

                # Check: no note on ANY string occupies any x_pos strictly between lx and rx
                if any(lx < occupied < rx for occupied in all_occupied_x) or random.random() >= 0.20:
                    continue

                x_left = (left_note['bbox'][0] + left_note['bbox'][2]) / 2
                x_right = (right_note['bbox'][0] + right_note['bbox'][2]) / 2
                y_anchor = max(left_note['bbox'][3], right_note['bbox'][3]) + int(self.line_spacing) * random.uniform(0.05, 0.25)

                arc_depth = self.line_spacing * random.uniform(0.3, 0.75)
                cx_ctrl, cy_ctrl = (x_left + x_right) / 2, y_anchor + arc_depth

                arc_pts = []
                for t in np.linspace(0, 1, 30):
                    bx = (1 - t) ** 2 * x_left + 2 * (1 - t) * t * cx_ctrl + t ** 2 * x_right
                    by = (1 - t) ** 2 * y_anchor + 2 * (1 - t) * t * cy_ctrl + t ** 2 * y_anchor
                    arc_pts.append((bx, by))

                draw_final.line(arc_pts, fill=bottom_arcs_color, width=arc_width)



    def _draw_notes_and_get_labels(self, draw_final, width, height, stroke_width):
        """
        DRAWING NOTES ON TOP LAYER, APPLYING SLIDE EFFECTS, AND GENERATING LABELS
        Renders individual fret numbers, draws slide lines randomly, and prepares
        bounding box arrays for Model A (DCRN) and sequence sequences for Model B (CRNN).
        """
        labels_model_a = []
        chords_model_b = {}

        for note in self.notes:
            x = note['x_pos'] * self.chord_spacing
            y = note['render_y']
            text = str(note['fret'])
            bbox = note['bbox']
            string_num = note['string']

            # Draw the digit
            draw_final.text((x, y), text, fill=self.numbers_color, font=self.font, anchor="mm", stroke_width=stroke_width, stroke_fill=self.numbers_color)

            # Draw Slide In effect randomly
            if random.random() < self.add_elem_percentage/8:
                slide_width = max(1, 5)
                slide_color = tuple(max(0, min(255, c + random.randint(-3, 3))) for c in self.line_color)
                y_offset = random.randint(-7, 7)
                if random.random() < 0.5:  # Slide up
                    draw_final.line([
                        (bbox[0] - self.line_spacing * 0.3 + random.randint(-15, 7), y + self.line_spacing * 0.25 + y_offset),
                        (bbox[0] - self.line_spacing * 0.05 + random.randint(-15, 0), y - self.line_spacing * 0.25 + y_offset)
                    ], fill=slide_color, width=max(1, slide_width))
                else:  # Slide down
                    draw_final.line([
                        (bbox[2] + self.line_spacing * 0.05 + random.randint(0, 15), y - self.line_spacing * 0.25 + y_offset),
                        (bbox[2] + self.line_spacing * 0.3 + random.randint(-7, 15), y + self.line_spacing * 0.25 + y_offset)
                    ], fill=slide_color, width=max(1, slide_width))

            # Label generation for DCRN (Model A: Bounding boxes using YOLO format)
            class_id = 25 if str(note['fret']) == 'X' else note['fret']
            w_box = (bbox[2] - bbox[0]) / width
            h_box = (bbox[3] - bbox[1]) / height
            x_center = x / width
            y_center = y / height
            labels_model_a.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}")

            # Label generation for CRNN (Model B: Sequence recognition)
            chords_model_b.setdefault(x, []).append(f"digit.{note['fret']}:{string_num}")

        # Assemble sequence format
        sorted_chords_model_b = sorted(chords_model_b.keys())
        sequence_steps_model_b = [" ".join(str(f) for f in chords_model_b[x]) for x in sorted_chords_model_b]
        labels_model_b = " + ".join(sequence_steps_model_b)

        return labels_model_a, labels_model_b



    def _draw_time_signatures(self, draw_final, string_y_positions):
        """Randomly draws a time signature (like 4/4) at the start of the staff."""
        if random.random() > self.time_elem_percentage*2: return

        if getattr(self, 'has_time_sig', False):
            ts_top = random.choice(["2", "3", "4", "5", "6", "7", "9", "12"])
            ts_bottom = random.choice(["4", "8"])
            ts_size = self.font_size * random.uniform(1.2, 2.8)
            try:
                ts_font = ImageFont.truetype(random.choice(self.fonts), ts_size)
            except:
                ts_font = ImageFont.truetype("arialbd.ttf", ts_size)
            # Draw exactly in the reserved 1st column area
            x_pos = 1 * self.chord_spacing
            y_center_top = (string_y_positions[0] + string_y_positions[2]) / 2
            y_center_bottom = (string_y_positions[3] + string_y_positions[5]) / 2
            draw_final.text((x_pos, y_center_top), ts_top, fill=self.numbers_color, font=ts_font, anchor="mm")
            draw_final.text((x_pos, y_center_bottom), ts_bottom, fill=self.numbers_color, font=ts_font, anchor="mm")



    def _draw_rests(self, draw_final, string_y_positions):
        """Draws professional musical rests using the Segoe UI Symbol font."""
        if random.random() > self.add_elem_percentage: return

        empty_cols = getattr(self, 'empty_columns_for_rests', [])
        for col_x in empty_cols:
            x_pos = col_x * self.chord_spacing
            rest_color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in self.numbers_color)
            
            # \U0001D13B (whole), \U0001D13C (half), \U0001D13D (quarter), \U0001D13E (eighth), \U0001D13F (sixteenth)
            rest_char = random.choice(['\U0001D13B', '\U0001D13C', '\U0001D13D', '\U0001D13E', '\U0001D13F'])
            
            r_size = self.font_size * random.uniform(1.3, 1.9)
            try:
                # Windows system font that contains musical symbols
                r_font = ImageFont.truetype("seguisym.ttf", int(r_size))
            except:
                try:
                    r_font = ImageFont.truetype("arial.ttf", int(r_size))
                except:
                    continue
            
            # Position around the middle of the staff
            y_idx = random.choice([1, 2, 3])
            y_pos = string_y_positions[y_idx]
            
            draw_final.text((x_pos, y_pos), rest_char, fill=rest_color, font=r_font, anchor="mm")



    def _draw_time_line(self, draw_final, max_x, string_y_positions):
        """Draws a vertical line representing the current playback time."""
        if random.random() > self.time_elem_percentage: return

        time_x = random.randint(int(self.chord_spacing), int(max_x * self.chord_spacing))
        y_top = string_y_positions[0] - self.line_spacing * random.uniform(0.5, 3.0)
        y_bottom = string_y_positions[5] + self.line_spacing * random.uniform(0.5, 3.0)

        width = max(3, self.line_width + random.randint(0, 6))

        time_line_range = random.choice([(58, 255, 48), (185, 194, 192), (198, 216, 201)])
        time_line_color = time_line_range[0] + random.randint(-3, 3), time_line_range[1] + random.randint(-3, 3), time_line_range[2] + random.randint(-3, 3)

        draw_final.line([(time_x, y_top), (time_x, y_bottom)], fill=time_line_color, width=width)



    def _apply_noise(self, image):
        try:
            if random.random() < self.noise_percentage:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            if random.random() < self.noise_percentage:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            if random.random() < self.noise_percentage:
                np_image = np.array(image)
                noise = np.random.normal(loc=0, scale=15, size=np_image.shape)
                np_image = np.clip(np_image.astype('int16') + noise, 0, 255).astype('uint8')
                image = Image.fromarray(np_image)

            if self.noise_percentage > 0.3:
                if random.random() < self.noise_percentage:
                    np_image = np.array(image)
                    kernel_size = random.choice([1, 3, 5, 7])
                    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
                    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                    kernel_motion_blur /= kernel_size
                    np_image = cv2.filter2D(np_image, -1, kernel_motion_blur)
                    image = Image.fromarray(np_image)

                if random.random() < self.noise_percentage:
                    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

                if random.random() < self.noise_percentage:
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=random.randint(10, 70))
                    buffer.seek(0)
                    image = Image.open(buffer).convert('RGB')

        except Exception as e:
            print(f"Error in _apply_noise: {e}")

        return image



    def _save_files(self, image, output_name, labels_model_a, labels_model_b):
        try:
            image.save(f"{self.data_dir}\\{output_name}.png")
            with open(f"{self.data_dir}\\{output_name}_a.txt", "w") as f:
                f.write("\n".join(labels_model_a))
            with open(f"{self.data_dir}\\{output_name}_b.txt", "w") as f:
                f.write(labels_model_b)
        except Exception as e:
            print(f"Error in save_files: {e}")





if __name__ == "__main__":
    generators = {
        NotesGenerator('..\\train_set\\noiseH', noise_percentage=0.6),
        NotesGenerator('..\\train_set\\noiseM', noise_percentage=0.3),
        NotesGenerator('..\\train_set\\noiseL', noise_percentage=0.0),
        NotesGenerator('..\\train_set\\addH', add_elem_percentage=0.8),
        NotesGenerator('..\\train_set\\addM', add_elem_percentage=0.4),
        NotesGenerator('..\\train_set\\addL', add_elem_percentage=0.0),
        NotesGenerator('..\\train_set\\timeH', time_elem_percentage=0.4),
        NotesGenerator('..\\train_set\\timeM', time_elem_percentage=0.2),
        NotesGenerator('..\\train_set\\timeL', time_elem_percentage=0.0),
    }

    for generator in generators:
        print(f"Starting the generation of {GEN_NUMBER} images in {generator.data_dir}...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(generator.generate_single_tab, range(GEN_NUMBER)), total=GEN_NUMBER))

        print("\nGeneration completed!\n\n")