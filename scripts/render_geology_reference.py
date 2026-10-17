"""Render the coordinate-bearing regional PDF using bundled PDF libraries."""
import json
from pathlib import Path
import pypdfium2 as pdfium
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'data/pilot_nk/references/North-Central_2025.pdf'
out=ROOT/'outputs/udegi/reconnaissance'
page=PdfReader(path).pages[0]
vp=page['/VP'][0]
measure=vp['/Measure'].get_object()
gcs=measure['/GCS'].get_object()
print(gcs)
doc=pdfium.PdfDocument(str(path))
image=doc[0].render(scale=2).to_pil()
x0,y0,x1,y1=map(float,vp['/BBox'])
height=float(page.mediabox.height)
box=(round(min(x0,x1)*2),round((height-max(y0,y1))*2),
     round(max(x0,x1)*2),round((height-min(y0,y1))*2))
image.crop(box).save(out/'regional_geology_map.png')
image.resize((1993,1123)).save(out/'regional_geology_full.png')
(out/'regional_geology_coordinates.json').write_text(json.dumps(dict(
    source=str(path),viewport_bbox=list(map(float,vp['/BBox'])),
    geographic_points=list(map(float,measure['/GPTS'])),
    local_points=list(map(float,measure['/LPTS'])),gcs=str(gcs),scale='1:1,000,000'),indent=2))
