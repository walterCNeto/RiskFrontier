from PIL import Image

# Troque o nome abaixo se o seu arquivo original tiver outro nome
src = "logoEnf.jpg"   # pode apontar para .png também
dst_pdf = "logoEnf.pdf"
dst_png = "logoEnf_fixed.png"

im = Image.open(src).convert("RGB")  # garante espaço de cor sRGB
im.save(dst_pdf)                     # gera PDF (raster)
im.save(dst_png, optimize=True)      # gera PNG válido

print("Gerado:", dst_pdf, "e", dst_png)
