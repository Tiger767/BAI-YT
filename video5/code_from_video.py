from image import *

x1 = load('corn_field.jpg', (512, 512))
x2 = load('sunset.jpg', (512, 512))

ws = Windows()
ws.start()
cf = ws.add('corn_field', x1)
ss = ws.add('sunset', x2)

ws.set(cf, gray(x1))
ws.set(ss, resize(x2, (64, 64)))
ws.set(cf, pyr(x1, -1))
save('modified_corn_field.jpg', increase_brightness(x1, 50, relative=True))
ws.set(cf, set_brightness(x1, 10))
ws.set(cf, set_gamma(x1, 2.0))
ws.set(ss, gray(x2))
ws.set(ss, apply_clahe(gray(x2)))
ws.set(ss, apply_clahe(x2))
ws.set(ss, equalize(gray(x2)))
ws.set(cf, rotate(x1, 45))
ws.set(cf, translate(x1, 50, 100))
ws.set(ss, crop_rect(x2, 10, 20, 40, 80))
ws.set(ss, shrink_sides(x2, 20, 20, 20, 20))
ws.set(ss, pad(x2, 20, 20, 20, 20))
img = ws.add(image=blend(x1, x2, .7))
ws.set(cf, zoom(x1, (64, 64), 0, 0))
ws.stop()
