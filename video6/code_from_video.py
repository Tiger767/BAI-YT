from audio import *


x, r, at = record(1, 16000)

play(x, r)

calc_duration(x, r)

play(set_length(x, 8000), r)

play(set_duration(x, r, 2), r)

for_each_frame(x, r, .1, calc_rms)

spectro, spectro_r = compute_spectrogram(x, r, .1)

play(*convert_spectrogram_to_audio(spectro, spectro_r))

play(shift_pitch(x, r, 100), r)

play(set_power(x, calc_rms(x)/2), r)

play(adjust_speed(x, r, 2), r)

play(set_speed(x, r, 2), r)

play(adjust_volume(x, 2), r)




