from wand.image import Image

pdf = Image(filename='./cad2.pdf', resolution=300)

pdfImage = pdf.convert("jpeg")

img = pdfImage.sequence[0]

page = Image(image=img)
page.save(filename="cad3.jpg")
