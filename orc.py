from PIL import Image
import pytesseract

img = Image.open('./cad1.jpg')
text = pytesseract.image_to_string(img)
print(text)