import argparse
import math
import wave
import numpy as np
import pyaudio
import pygame
import params
import midi_utils
import keras
from keras.models import Model, load_model
from keras import backend as K
dir_name = ''
sub_dir_name = ''
sample_rate = 48000
note_dt = 2000  # num samples
note_duration = 20000  # num samples
note_decay = 5.0 / sample_rate
num_params = params.num_params
num_measures = 16
num_sigmas = 5.0
note_threshold = 32
use_pca = True
is_ae = True
autosave = False
autosavenum = 1
autosavenow = False
blend = False
blendfactor = np.float32(1.0)
blendstate = 0
background_color = (210, 210, 210)
edge_color = (60, 60, 60)
slider_colors = [(90, 20, 20), (90, 90, 20), (20, 90, 20),
                 (20, 90, 90), (20, 20, 90), (90, 20, 90)]

note_w = 96
note_h = 96
note_pad = 2

notes_rows = int(num_measures / 8)
notes_cols = 8

slider_num = min(40, num_params)
slider_h = 200
slider_pad = 5
tick_pad = 4

control_w = 200
control_h = 30
control_pad = 5
control_num = 5
control_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]
control_inits = [0.75, 0.5, 0.5, 0.5, 0.5]
notes_w = notes_cols * (note_w + note_pad * 2)
notes_h = notes_rows * (note_h + note_pad * 2)
sliders_w = notes_w
sliders_h = slider_h + slider_pad * 2
controls_w = control_w * control_num
controls_h = control_h
window_w = max(notes_w, controls_w)
window_h = notes_h + sliders_h + controls_h
slider_w = int((window_w - slider_pad * 2) / slider_num)
notes_x = 0
notes_y = sliders_h
text_x = notes_w + 5
text_y = notes_y + 5
text_h = 40
text_w = 200
sliders_x = slider_pad
sliders_y = slider_pad
controls_x = int((window_w - controls_w) / 2)
controls_y = notes_h + sliders_h
keyframe_paths = np.array(("song 1.txt", "song 2.txt", ))
prev_mouse_pos = None
mouse_pressed = 0
cur_slider_ix = 0
cur_control_ix = 0
volume = 3000
balance = 0.5
instrument = 0
needs_update = True
current_params = np.zeros((num_params,), dtype=np.float32)
keyframe_params = np.zeros((len(keyframe_paths),num_params),dtype=np.float32)
current_notes = np.zeros((num_measures, note_h, note_w), dtype=np.uint8)
cur_controls = np.array(control_inits, dtype=np.float32)
keyframe_controls = np.zeros((len(keyframe_paths),len(cur_controls)),dtype=np.float32)
blend_slerp = False
keyframe_magnitudes = np.zeros((len(keyframe_paths),),dtype=np.float32)
songs_loaded = False
audio = pyaudio.PyAudio()
audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False
def audio_callback(in_data, frame_count, time_info, status):
    """
    Audio call-back to influence playback of music with input.
    :param in_data:
    :param frame_count:
    :param time_info:
    :param status:
    :return:
    """
    global audio_time
    global audio_notes
    global audio_reset
    global note_time
    global note_time_dt
    global autosavenow
    global autosave
    global audio_pause
    global blendstate
    global blendfactor
    global keyframe_paths
    global balance

    # check if needs restart
    if audio_reset:
        audio_notes = []
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_reset = False

    # check if paused
    if audio_pause and status is not None:
        data = np.zeros((frame_count,), dtype=np.float32)
        return data.tobytes(), pyaudio.paContinue

    # find and add any notes in this time window
    cur_dt = note_dt
    while note_time_dt < audio_time + frame_count:
        measure_ix = int(note_time / note_h)
        if measure_ix >= num_measures:
            break
        note_ix = note_time % note_h
        notes = np.where(
            current_notes[measure_ix, note_ix] >= note_threshold)[0]
        for note in notes:
            freq = 2 * 38.89 * pow(2.0, note / 12.0) / sample_rate
            audio_notes.append((note_time_dt, freq, current_notes[measure_ix, note_ix, note]))
        note_time += 1
        note_time_dt += cur_dt

    # generate the tones
    data = np.zeros((frame_count,), dtype=np.float32)
    for t, f, v in audio_notes:
        x = np.arange(audio_time - t, audio_time + frame_count - t)
        x = np.maximum(x, 0)

        if instrument == 0:
            w = np.sign(1 - np.mod(x * f, 2))  # Square
        elif instrument == 1:
            w = np.mod(x * f - 1, 2) - 1  # Sawtooth
        elif instrument == 2:
            w = 2 * np.abs(np.mod(x * f - 0.5, 2) - 1) - 1  # Triangle
        elif instrument == 3:
            w = np.sin(x * f * math.pi)  # Sine
        elif instrument == 4:
            w = -1 * np.sign(np.mod(2*x*f,4)-2) * np.sqrt( 1-( ( np.mod(2*x*f,2)-1) *( ( np.mod(2*x*f,2)-1) ) ))  # Circle

        # w = np.floor(w*8)/8
        w[x == 0] = 0
        n = 12 * np.log (f * sample_rate / 38.89) / np.log(2);
        w *= volume * np.exp(-x * note_decay) * pow(balance, (n - 60) / 12.0) / np.log(2)
        if params.encode_volume:
            w *= v / 255
        data += w
    data = np.clip(data, -32000, 32000).astype(np.int16)

    # remove notes that are too old
    audio_time += frame_count
    audio_notes = [(t, f, v)
                   for t, f, v in audio_notes if audio_time < t + note_duration]
    blendfactor = (np.cos( ((note_time / note_h)/num_measures) * math.pi )+1)/2
    #print(blendfactor)
    # reset if loop occurs
    if note_time / note_h >= num_measures:
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_notes = []
        blendstate = (blendstate+1)%(2*len(keyframe_paths))
        #if blendstate == 0:
            #audio_pause = True
        blendfactor = 1
        if autosave and not autosavenow:
            autosavenow = True

    # return the sound clip
    return data.tobytes(), pyaudio.paContinue


def update_mouse_click(mouse_pos):
    """
    Update control stated based on where the mouse clicked.
    :param mouse_pos:
    :return:
    """
    global cur_slider_ix
    global cur_control_ix
    global mouse_pressed
    x = (mouse_pos[0] - sliders_x)
    y = (mouse_pos[1] - sliders_y)

    if 0 <= x < sliders_w and 0 <= y < sliders_h:
        cur_slider_ix = int(x / slider_w)
        mouse_pressed = 1

    x = (mouse_pos[0] - controls_x)
    y = (mouse_pos[1] - controls_y)
    if 0 <= x < controls_w and 0 <= y < controls_h:
        cur_control_ix = int(x / control_w)
        mouse_pressed = 2


def apply_controls():
    """
    Change parameters based on controls.
    :return:
    """
    global note_threshold
    global note_dt
    global volume
    global note_duration
    global note_decay
    global sample_rate
    global balance

    note_threshold = (1.0 - cur_controls[0]) * 200 + 10
    note_dt = (1.0 - cur_controls[1]) * 1800 + 200
    volume = cur_controls[2] * 6000
    balance = pow(2, cur_controls[3] * 4 - 2);

    note_duration = 10000 / ((1-cur_controls[4]) + 0.001)
    note_decay = 10 * (1 - cur_controls[4]) / sample_rate


def update_mouse_move(mouse_pos):
    """
    Update sliders/controls based on mouse input.
    :param mouse_pos:
    :return:
    """
    global needs_update
    t = 1
    if int(cur_control_ix) == 0:
        t = 210.0 / 200
    if mouse_pressed == 1:
        # change sliders
        y = (mouse_pos[1] - sliders_y)
        if 0 <= y <= slider_h:
            val = (float(y) / slider_h - 0.5) * (num_sigmas * 2)
            current_params[int(cur_slider_ix)] = val
            needs_update = True
    elif mouse_pressed == 2:
        # change controls
        x = (mouse_pos[0] - (controls_x + cur_control_ix * control_w))
        if control_pad <= x <= control_w - control_pad:
            val = float(x - control_pad) / (control_w - control_pad * 2)
            cur_controls[int(cur_control_ix)] = val * t
            apply_controls()


def draw_controls(screen):
    """
    Draw volume and threshold controls to screen.
    :param screen:
    :return:
    """
    #allows for higher threshold
    t = 200.0 / 210
    for i in range(control_num):
        x = controls_x + i * control_w + control_pad
        y = controls_y + control_pad
        w = control_w - control_pad * 2
        h = control_h - control_pad * 2
        col = control_colors[i]

        pygame.draw.rect(screen, col, (x, y, int(w * t * cur_controls[i]), h))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, w, h), 1)

        t = 1


def draw_sliders(screen):
    """
    Draw sliders to screen.
    :param screen:
    :return:
    """
    for i in range(slider_num):
        slider_color = slider_colors[i % len(slider_colors)]
        x = sliders_x + i * slider_w
        y = sliders_y

        cx = x + slider_w / 2
        cy_1 = y
        cy_2 = y + slider_h
        pygame.draw.line(screen, slider_color, (cx, cy_1), (cx, cy_2))

        cx_1 = x + tick_pad
        cx_2 = x + slider_w - tick_pad
        for j in range(int(num_sigmas * 2 + 1)):
            ly = y + slider_h / 2.0 + \
                (j - num_sigmas) * slider_h / (num_sigmas * 2.0)
            ly = int(ly)
            col = (0, 0, 0) if j - num_sigmas == 0 else slider_color
            pygame.draw.line(screen, col, (cx_1, ly), (cx_2, ly))

        py = y + int((current_params[i] / (num_sigmas * 2) + 0.5) * slider_h)
        pygame.draw.circle(screen, slider_color, (int(
            cx), int(py)), int((slider_w - tick_pad) / 2))


def get_pianoroll_from_notes(notes):
    """
    Draw piano roll of notes.
    :param notes:
    :return:
    """
    
    output = np.full((3, int(notes_h), int(notes_w)), 64, dtype=np.uint8)

    for i in range(notes_rows):
        for j in range(notes_cols):
            x = note_pad + j * (note_w + note_pad * 2)
            y = note_pad + i * (note_h + note_pad * 2)
            ix = i * notes_cols + j

            measure = np.rot90(notes[ix])

            played_only = np.where(measure >= note_threshold, 255, 0)
            output[0, y:y + note_h, x:x +
                   note_w] = np.minimum(measure * (255.0 / note_threshold), 255.0)
            output[1, y:y + note_h, x:x + note_w] = played_only
            output[2, y:y + note_h, x:x + note_w] = played_only

    return np.transpose(output, (2, 1, 0))


def draw_notes(screen, notes_surface):
    """
    Draw pianoroll notes to screen.
    :param screen:
    :param notes_surface:
    :return:
    """

    pygame.surfarray.blit_array(
        notes_surface, get_pianoroll_from_notes(current_notes))

    measure_ix = int(note_time / note_h)
    note_ix = note_time % note_h
    x = notes_x + note_pad + (measure_ix % notes_cols) * \
        (note_w + note_pad * 2) + note_ix
    y = notes_y + note_pad + \
        int(measure_ix / notes_cols) * (note_h + note_pad * 2)

    pygame.draw.rect(screen, (255, 255, 0), (x, y, 4, note_h), 0)


def play():
    global mouse_pressed
    global current_notes
    global audio_pause
    global needs_update
    global current_params
    global prev_mouse_pos
    global audio_reset
    global instrument
    global songs_loaded
    global autosavenow
    global autosavenum
    global autosave
    global blend
    global blendstate
    global blendfactor
    global keyframe_params
    global keyframe_controls
    global keyframe_paths
    global cur_controls
    global keyframe_magnitudes
    global blend_slerp

    print("Keras version: " + keras.__version__)

    K.set_image_data_format('channels_first')

    print("Loading encoder...")
    model = load_model(dir_name + 'model.h5')
    encoder = Model(inputs=model.input,
                    outputs=model.get_layer('encoder').output)
    decoder = K.function([model.get_layer('decoder').input, K.learning_phase()],
                         [model.layers[-1].output])

    print("Loading gaussian/pca statistics...")
    latent_means = np.load(dir_name + sub_dir_name + '/latent_means.npy')
    latent_stds = np.load(dir_name + sub_dir_name + '/latent_stds.npy')
    latent_pca_values = np.load(
        dir_name + sub_dir_name + '/latent_pca_values.npy')
    latent_pca_vectors = np.load(
        dir_name + sub_dir_name + '/latent_pca_vectors.npy')

    # open a window
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((int(window_w), int(window_h)))
    notes_surface = screen.subsurface((notes_x, notes_y, notes_w, notes_h))
    pygame.display.set_caption('Neural Composer')

    # start the audio stream
    audio_stream = audio.open(
        format=audio.get_format_from_width(2),
        channels=1,
        rate=sample_rate,
        output=True,
        stream_callback=audio_callback)
    audio_stream.start_stream()

    # main loop
    running = True
    random_song_ix = 0
    cur_len = 0
    blendcycle = 0
    apply_controls()
    while running:
        # process events
        if autosavenow:
            # generate random song
            current_params = np.clip(np.random.normal(
                0.0, 1.0, (num_params,)), -num_sigmas, num_sigmas)
            needs_update = True
            audio_reset = True
            # save slider values
            with open("results/history/autosave" + str(autosavenum)+".txt", "w") as text_file:
                text_file.write(sub_dir_name + "\n")
                text_file.write(str(instrument) + "\n")
                for iter in cur_controls:
                    text_file.write(str(iter) + "\n")
                for iter in current_params:
                    text_file.write(str(iter) + "\n")
            # save song as wave
            audio_pause = True
            audio_reset = True
            save_audio = b''
            while True:
                save_audio += audio_callback(None, 1024, None, None)[0]
                if audio_time == 0:
                    break
            wave_output = wave.open('results/history/autosave' + str(autosavenum)+'.wav', 'w')
            wave_output.setparams(
                (1, 2, sample_rate, 0, 'NONE', 'not compressed'))
            wave_output.writeframes(save_audio)
            wave_output.close()
            audio_pause = False
            autosavenum += 1
            autosavenow = False
            needs_update = True
            audio_reset = True
        blendcycle += 1
        if blend and blendcycle > 10:
            blendcycle = 0
            if blendstate%2 == 0:
                needs_update = True
                current_params = np.copy(keyframe_params[int(blendstate/2)])
                cur_controls = np.copy(keyframe_controls[int(blendstate/2)])
                apply_controls()
            elif blendstate%2 == 1:
                for x in range(0,len(current_params)):
                    current_params[x] = (blendfactor * keyframe_params[int(blendstate/2),x]) + ((1-blendfactor)*keyframe_params[((int(blendstate/2))+1)%len(keyframe_paths),x])
                if blend_slerp:
                    magnitude = (blendfactor * keyframe_magnitudes[int(blendstate/2)]) + ((1-blendfactor)*keyframe_magnitudes[((int(blendstate/2))+1)%len(keyframe_paths)])
                    current_params = current_params * ((sum(current_params*current_params)**-0.5) * magnitude)
                for x in range(0,len(cur_controls)):
                    cur_controls[x] = (blendfactor * keyframe_controls[int(blendstate/2),x]) + ((1-blendfactor)*keyframe_controls[((int(blendstate/2))+1)%len(keyframe_paths),x])
                apply_controls()
                needs_update = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # QUIT BUTTON HIT
                running = False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:  # MOUSE BUTTON DOWN
                if pygame.mouse.get_pressed()[0]:
                    prev_mouse_pos = pygame.mouse.get_pos()
                    update_mouse_click(prev_mouse_pos)
                    update_mouse_move(prev_mouse_pos)
                elif pygame.mouse.get_pressed()[2]:
                    current_params = np.zeros((num_params,), dtype=np.float32)
                    needs_update = True

            elif event.type == pygame.MOUSEBUTTONUP:   # MOUSE BUTTON UP
                mouse_pressed = 0
                prev_mouse_pos = None

            elif event.type == pygame.MOUSEMOTION and mouse_pressed > 0:  # MOUSE MOTION WHILE PRESSED
                update_mouse_move(pygame.mouse.get_pos())

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # KEYDOWN R
                    # generate random song
                    current_params = np.clip(np.random.normal(
                        0.0, 1.0, (num_params,)), -num_sigmas, num_sigmas)
                    needs_update = True
                    audio_reset = True
                if event.key == pygame.K_t:  # KEYDOWN T
                    for x in range(int(num_params/3)+1, num_params):
                        current_params[x] = np.clip(np.random.normal(0.0,1.0), -num_sigmas, num_sigmas)
                    needs_update = True
                if event.key == pygame.K_x:  # KEYDOWN X
                    # generate random song
                    current_params += np.clip(np.random.normal(
                        0.0, 0.3, (num_params,)), -num_sigmas, num_sigmas)
                    needs_update = True
                if event.key == pygame.K_a:  # KEYDOWN A
                    autosave = not autosave
                if event.key == pygame.K_b:  # KEYDOWN B
                    blend = not blend
                    blendstate = 0
                    blendfactor = 1.0
                    if blend:
                        audio_pause = True
                        audio_reset = True
                        needs_update = True
                        blendnum = int(input("The number of songs to be blended "))
                        keyframe_paths = []
                        keyframe_controls = np.zeros((blendnum,len(cur_controls)),dtype=np.float32)
                        keyframe_params = np.zeros((blendnum,num_params),dtype=np.float32)
                        for y in range(blendnum):
                            fileName = input("The file name of the next song to be blended ")
                            if "." not in fileName:
                                fileName = fileName + ".txt"
                            keyframe_paths.append((fileName))
                            fo = open("results/history/" + fileName, "r")
                            if not sub_dir_name == fo.readline()[:-1]:
                                running = false
                                print("incompatable with current model")
                                break
                            instrument = int(fo.readline())
                            for x in range(len(cur_controls)):
                                keyframe_controls[y,x] = float(fo.readline())
                            for x in range(len(current_params)):
                                keyframe_params[y,x] = float(fo.readline())
                            #keyframe_magnitudes[y] = sum(keyframe_params[y]*keyframe_params[y])**0.5
                if event.key == pygame.K_e:  # KEYDOWN E
                    # generate random song with larger variance
                    current_params = np.clip(np.random.normal(0.0, 2.0, (num_params,)), -num_sigmas, num_sigmas)
                    needs_update = True
                    audio_reset = True
                if event.key == pygame.K_PERIOD:
                    current_params /= 1.1
                    needs_update = True
                if event.key == pygame.K_COMMA:
                    current_params *= 1.1
                    needs_update = True
                if event.key == pygame.K_SLASH:
                    current_params *= -1
                    needs_update = True
                if event.key == pygame.K_UP:
                    cur_controls[0] = (210.0 - note_threshold + 1) / 200
                    apply_controls()
                if event.key == pygame.K_DOWN:
                    cur_controls[0] = (210.0 - note_threshold - 1) / 200
                    apply_controls()
                if event.key == pygame.K_s:  # KEYDOWN S
                    # save slider values
                    audio_pause = True
                    fileName = input("File Name to save into ")
                    if "." not in fileName:
                        fileName = fileName + ".txt"
                    with open("results/history/" + fileName, "w") as text_file:
                        if blend:
                            text_file.write(sub_dir_name + "\n")
                            text_file.write("blended song" + "\n")
                            text_file.write(str(len(keyframe_paths)) + "\n")
                            for x in range(len(keyframe_paths)):
                                text_file.write("" + keyframe_paths[x] + "\n")
                        else:
                            text_file.write(sub_dir_name + "\n")
                            text_file.write(str(instrument) + "\n")
                            for iter in cur_controls:
                                text_file.write(str(iter) + "\n")
                            for iter in current_params:
                                text_file.write(str(iter) + "\n")
                if event.key == pygame.K_l:  # KEYDOWN L
                    audio_pause = True
                    needs_update = True
                    audio_reset = True
                    fileName = input("File Name to read ")
                    if "." not in fileName:
                        fileName = fileName + ".txt"
                    fo = open("results/history/" + fileName, "r")
                    print (fo.name)
                    if not sub_dir_name == fo.readline()[:-1]:
                                running = false
                                print("incompatable with current model")
                                break
                    tempDir = fo.readline()
                    if tempDir.startswith("blended song"):
                        blend = True
                        blendnum = int(fo.readline())
                        keyframe_paths = []
                        keyframe_controls = np.zeros((blendnum,len(cur_controls)),dtype=np.float32)
                        keyframe_params = np.zeros((blendnum,num_params),dtype=np.float32)
                        for y in range(blendnum):
                            fileName2 = fo.readline()[:-1]
                            keyframe_paths.append(fileName)
                            fo2 = open("results/history/" + fileName2, "r")
                            if not sub_dir_name == fo2.readline()[:-1]:
                                running = false
                                print("incompatable with current model")
                                break
                            instrument = int(fo2.readline())
                            for x in range(len(cur_controls)):
                                keyframe_controls[y,x] = float(fo2.readline())
                            for x in range(len(current_params)):
                                keyframe_params[y,x] = float(fo2.readline())
                    else:
                        instrument = int(tempDir)
                        for x in range(len(cur_controls)):
                            cur_controls[x] = float(fo.readline())
                        for x in range(len(current_params)):
                            current_params[x] = float(fo.readline())
                    apply_controls()
                if event.key == pygame.K_o:  # KEYDOWN O

                    if not songs_loaded:
                        print("Loading songs...")
                        try:
                            y_samples = np.load('data/interim/samples.npy')
                            y_lengths = np.load('data/interim/lengths.npy')
                            songs_loaded = True
                        except Exception as e:
                            print("This functionality is to check if the model training went well by reproducing an original song. "
                                  "The composer could not load samples and lengths from model training. "
                                  "If you have the midi files, the model was trained with, process them by using"
                                  " the preprocess_songs.py to find the requested files in data/interim "
                                  "(Load exception: {0}".format(e))

                    if songs_loaded:
                        # check how well the autoencoder can reconstruct a random song
                        print("Random Song Index: " + str(random_song_ix))
                        if is_ae:
                            example_song = y_samples[cur_len:cur_len + num_measures]
                            current_notes = example_song * 255
                            latent_x = encoder.predict(np.expand_dims(
                                example_song, 0), batch_size=1)[0]
                            cur_len += y_lengths[random_song_ix]
                            random_song_ix += 1
                        else:
                            random_song_ix = np.array(
                                [random_song_ix], dtype=np.int64)
                            latent_x = encoder.predict(
                                random_song_ix, batch_size=1)[0]
                            random_song_ix = (
                                random_song_ix + 1) % model.layers[0].input_dim

                        if use_pca:
                            current_params = np.dot(
                                latent_x - latent_means, latent_pca_vectors.T) / latent_pca_values
                        else:
                            current_params = (
                                latent_x - latent_means) / latent_stds

                        needs_update = True
                        audio_reset = True

                if event.key == pygame.K_m:  # KEYDOWN M
                    # save song as midi
                    audio_pause = True
                    audio_reset = True
                    fileName = input("File Name to save into ")
                    if "." not in fileName:
                        fileName = fileName + ".mid"
                    midi_utils.samples_to_midi(
                        current_notes, 'results/history/' + fileName, note_threshold)
                    audio_pause = False

                if event.key == pygame.K_w:  # KEYDOWN W
                    # save song as wave
                    audio_pause = True
                    audio_reset = True
                    fileName = input("File Name to save into ")
                    if "." not in fileName:
                        fileName = fileName + ".wav"
                    save_audio = b''
                    while True:
                        save_audio += audio_callback(None, 1024, None, None)[0]
                        if audio_time == 0:
                            break
                    wave_output = wave.open('results/history/' + fileName + '.wav', 'w')
                    wave_output.setparams(
                        (1, 2, sample_rate, 0, 'NONE', 'not compressed'))
                    wave_output.writeframes(save_audio)
                    wave_output.close()
                    audio_pause = False

                if event.key == pygame.K_ESCAPE:  # KEYDOWN ESCAPE
                    # exit application
                    running = False
                    break

                if event.key == pygame.K_SPACE:  # KEYDOWN SPACE
                    # toggle pause/play audio
                    audio_pause = not audio_pause

                if event.key == pygame.K_TAB:  # KEYDOWN TAB
                    # reset audio playing
                    audio_reset = True
                    if autosave and not autosavenow:
                        autosavenow = True

                if event.key == pygame.K_1:  # KEYDOWN 1
                    # play instrument 0
                    instrument = 0

                if event.key == pygame.K_2:  # KEYDOWN 2
                    # play instrument 1
                    instrument = 1

                if event.key == pygame.K_3:  # KEYDOWN 3
                    # play instrument 2
                    instrument = 2

                if event.key == pygame.K_4:  # KEYDOWN 4
                    # play instrument 3
                    instrument = 3

                if event.key == pygame.K_5:  # KEYDOWN 5
                    # play instrument 4
                    instrument = 4

                if event.key == pygame.K_c:  # KEYDOWN C
                    #
                    y = np.expand_dims(
                        np.where(current_notes > note_threshold, 1, 0), 0)
                    latent_x = encoder.predict(y)[0]
                    if use_pca:
                        current_params = np.dot(
                            latent_x - latent_means, latent_pca_vectors.T) / latent_pca_values
                    else:
                        current_params = (
                            latent_x - latent_means) / latent_stds
                    needs_update = True

        # check if params were changed so that a new song should be generated
        if needs_update:
            if use_pca:
                latent_x = latent_means + \
                    np.dot(current_params * latent_pca_values,
                           latent_pca_vectors)
            else:
                latent_x = latent_means + latent_stds * current_params
            latent_x = np.expand_dims(latent_x, axis=0)
            y = decoder([latent_x, 0])[0][0]
            current_notes = (y * 255.0).astype(np.uint8)
            needs_update = False

        # draw GUI to the screen
        screen.fill(background_color)
        draw_notes(screen, notes_surface)
        draw_sliders(screen)
        draw_controls(screen)

        # flip the screen buffer
        pygame.display.flip()
        pygame.time.wait(10)

    # if app is exited, close the audio stream
    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()


if __name__ == "__main__":
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(
        description='Neural Composer: Play and edit music of a trained model.')
    parser.add_argument('--model_path', type=str,
                        help='The folder the model is stored in (e.g. a folder named e and a number located in results/history/).', required=True)

    args = parser.parse_args()
    if args.model_path.endswith(".txt"):
        fo = open("results/history/" + args.model_path, "r")
        print (fo.name)
        sub_dir_name = fo.readline()[:-1]
        tempDir = fo.readline()
        if tempDir.startswith("blended song"):
            blend = True
            blendnum = int(fo.readline())
            keyframe_paths = []
            keyframe_controls = np.zeros((blendnum,len(cur_controls)),dtype=np.float32)
            keyframe_params = np.zeros((blendnum,num_params),dtype=np.float32)
            for y in range(blendnum):
                fileName2 = fo.readline()[:-1]
                keyframe_paths.append(fileName2)
                fo2 = open("results/history/" + fileName2, "r")
                if not sub_dir_name == fo2.readline()[:-1]:
                    running = false
                    print("incompatable with current model")
                    break
                instrument = int(fo2.readline())
                for x in range(len(cur_controls)):
                    keyframe_controls[y,x] = float(fo2.readline())
                for x in range(len(current_params)):
                    keyframe_params[y,x] = float(fo2.readline())
        else:
            print(sub_dir_name)
            instrument = int(tempDir)
            for x in range(len(cur_controls)):
                cur_controls[x] = float(fo.readline())
            for x in range(len(current_params)):
                current_params[x] = float(fo.readline())
        
    else:
        sub_dir_name = args.model_path
    play()
